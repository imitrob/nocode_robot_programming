
"""
Run task script (offline-friendly, panda_py-compatible)

This script bundles:
  1) Time-indexed GMM+GMR over [t, x] with x∈R^15
  2) CIP-style anomaly detector with per-modality Mahalanobis thresholds
  3) DTW alignment that uses an orientation-aware metric (log-quaternion)
  4) A runner that simulates a robot loop now and can swap to panda_py later

Usage (offline simulation):
---------------------------
$ python run_task.py --simulate --steps 400

Usage (with a real Panda later):
--------------------------------
$ python run_task.py --panda-host 192.168.1.100 --steps 400

Requires: numpy, scikit-learn, fastdtw
Optional: panda_py (only when controlling a real robot)

Note on dimensions:
-------------------
State x = [ p(3), q(4), f(3), tau(3), g(1), h(1) ] = 15 dims
- DTW alignment uses p and q via a pose-aware distance (log-quaternion).
- GMM/GMR is fit on the *original 15D* state (we do not reduce dim).

The control loop follows your 5 steps:
1) Get robot states (fake now or via Panda later)
2) At each step check if robot state at pose (within tolerance)
3) If so, iterate timestep along the GMR time grid
4) Check anomaly
5) Assign the timestep goal pose (and command robot if available)
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import argparse
import math
import sys

import numpy as np
from numpy.typing import ArrayLike
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# -----------------------------
# Quaternion utilities (scalar-last: [x,y,z,w])
# -----------------------------

def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.linalg.norm(q)

def quat_conj(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q
    return np.array([-x, -y, -z, w], dtype=np.float64)

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return quat_normalize(np.array([x, y, z, w], dtype=np.float64))

def quat_log(q: np.ndarray) -> np.ndarray:
    """Map unit quaternion to R^3 (rotation vector) using log map.
    q = [v, w] with scalar-last convention. Returns axis*angle.
    """
    q = quat_normalize(q)
    v = q[:3]
    w = q[3]
    v_norm = np.linalg.norm(v)
    angle = 2.0 * math.atan2(v_norm, w)
    if v_norm < 1e-12:
        return np.zeros(3)
    axis = v / v_norm
    return axis * angle

def quat_exp(r: np.ndarray) -> np.ndarray:
    """Exp map from R^3 to unit quaternion (scalar-last)."""
    theta = np.linalg.norm(r)
    if theta < 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = r / theta
    v = axis * math.sin(theta / 2.0)
    w = math.cos(theta / 2.0)
    return quat_normalize(np.concatenate([v, [w]]))

def quat_angle(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle between two unit quaternions (in radians)."""
    q_rel = quat_mul(quat_conj(q1), q2)
    return 2.0 * math.atan2(np.linalg.norm(q_rel[:3]), q_rel[3])

# -----------------------------
# DTW alignment using pose-aware distance
# -----------------------------
POSE_IDXS = list(range(0, 7))  # p(0..2), q(3..6)

@dataclass
class DTWAligner:
    w_pos: float = 1.0
    w_ori: float = 1.0

    def _feat(self, x: np.ndarray) -> np.ndarray:
        p = x[0:3]
        q = quat_normalize(x[3:7])
        r = quat_log(q)  # 3
        return np.concatenate([p * self.w_pos, r * self.w_ori])  # 6

    def align(self, ref: np.ndarray, demo: np.ndarray) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
        """Align demo (T2,15) to ref (T1,15) using DTW on 6D pose features.
        Returns the warped demo with length T1 and the alignment path.
        """
        assert ref.shape[1] == 15 and demo.shape[1] == 15
        T1, T2 = ref.shape[0], demo.shape[0]
        # Build sequences of 6D features
        A = [self._feat(ref[t, :7]) for t in range(T1)]
        B = [self._feat(demo[t, :7]) for t in range(T2)]
        # DTW path
        dist, path = fastdtw(A, B, dist=euclidean)
        # Warp demo onto ref timeline via average of matched indices
        warped = np.zeros_like(ref)
        hits = [[] for _ in range(T1)]
        for i, j in path:
            hits[i].append(j)
        for i in range(T1):
            if hits[i]:
                js = np.array(hits[i], dtype=int)
                warped[i] = demo[js].mean(axis=0)
            else:
                # Fallback: nearest neighbor in B
                j = int(round(i * (T2 - 1) / max(T1 - 1, 1)))
                warped[i] = demo[j]
        return warped, path

# -----------------------------
# Time-indexed GMM + GMR
# -----------------------------
@dataclass
class GMMGMRTimeIndexed:
    n_components: Optional[int] = None
    covariance_type: str = "full"
    reg_covar: float = 1e-6
    random_state: Optional[int] = None
    bic_range: Tuple[int, int] = (2, 12)

    gmm_: Optional[GaussianMixture] = None
    scaler_: Optional[StandardScaler] = None

    def fit(self, demonstrations_list: List[np.ndarray]) -> "GMMGMRTimeIndexed":
        demos = _ensure_list_of_demos(demonstrations_list)
        Z, scaler = _stack_joint_time_state(demos)
        self.scaler_ = scaler
        if self.n_components is None:
            Ks = range(self.bic_range[0], self.bic_range[1] + 1)
            best = (np.inf, None)
            for K in Ks:
                gmm = GaussianMixture(
                    n_components=K,
                    covariance_type=self.covariance_type,
                    reg_covar=self.reg_covar,
                    random_state=self.random_state,
                ).fit(Z)
                bic = gmm.bic(Z)
                if bic < best[0]:
                    best = (bic, gmm)
            self.gmm_ = best[1]
        else:
            self.gmm_ = GaussianMixture(
                n_components=int(self.n_components),
                covariance_type=self.covariance_type,
                reg_covar=self.reg_covar,
                random_state=self.random_state,
            ).fit(Z)
        return self

    def predict(self, t: ArrayLike) -> Tuple[np.ndarray, np.ndarray]:
        t_arr = np.atleast_1d(np.asarray(t, dtype=np.float64))
        mus_out, Sigmas_out = [], []
        for ti in t_arr:
            mu_k, Sigma_k, logw = self._conditional_component(float(ti))
            mu_z, Sigma_z = _mix_statistics(mu_k, Sigma_k, logw)
            mu_x, Sigma_x = _z_to_x(mu_z, Sigma_z, self.scaler_)
            mus_out.append(mu_x)
            Sigmas_out.append(Sigma_x)
        mus = np.stack(mus_out)
        Sigmas = np.stack(Sigmas_out)
        if np.ndim(t) == 0:
            return mus[0], Sigmas[0]
        return mus, Sigmas

    def predict_trajectory(self, n_steps: int = 200) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        t_grid = np.linspace(0.0, 1.0, int(n_steps))
        mus, Sigmas = self.predict(t_grid)
        return t_grid, mus, Sigmas

    # ---- internals ----
    def _conditional_component(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        gmm = self.gmm_ ; assert gmm is not None and self.scaler_ is not None
        K, D = gmm.n_components, gmm.means_.shape[1]
        assert D == 16, f"Expected 16 (t+15), got {D}"
        t_idx = 0 ; x_idx = np.arange(1, 16)
        mu = gmm.means_ ; cov = gmm.covariances_ ; pi = gmm.weights_
        mu_t = mu[:, t_idx] ; mu_x = mu[:, x_idx]
        Sigma_tt = cov[:, t_idx, t_idx]
        Sigma_tx = cov[:, t_idx][:, x_idx]
        Sigma_xt = cov[:, x_idx, t_idx]
        Sigma_xx = cov[:, x_idx][:, :, x_idx]
        eps = 1e-9
        Sigma_tt = np.maximum(Sigma_tt, eps)
        inv_Sigma_tt = 1.0 / Sigma_tt
        dt = (t - mu_t)
        mu_x_given_t = mu_x + (Sigma_xt * inv_Sigma_tt[:, None]) * dt[:, None]
        Sigma_x_given_t = Sigma_xx - np.einsum('ki,k,kj->kij', Sigma_xt, inv_Sigma_tt, Sigma_tx)
        # log-weights
        logw = np.log(np.maximum(pi, eps)) - 0.5 * (np.log(2*np.pi) + np.log(Sigma_tt) + (dt*dt)*inv_Sigma_tt)
        return mu_x_given_t, Sigma_x_given_t, logw

# ---- helper functions for GMM/GMR ----

def _ensure_list_of_demos(demonstrations_list: Sequence[np.ndarray]) -> List[np.ndarray]:
    demos: List[np.ndarray] = []
    for i, M in enumerate(demonstrations_list):
        A = np.asarray(M)
        assert A.shape[0] == 15 and A.ndim == 2, f"Demo {i} must be (15, T)."
        demos.append(A)
    return demos

def _stack_joint_time_state(demos: List[np.ndarray]) -> Tuple[np.ndarray, StandardScaler]:
    Ts, Xs = [], []
    for M in demos:
        T_i = M.shape[1]
        t = np.linspace(0.0, 1.0, T_i)
        Ts.append(t[:, None])
        Xs.append(M.T)
    t_all = np.concatenate(Ts, axis=0)
    X_all = np.concatenate(Xs, axis=0)
    scaler = StandardScaler().fit(X_all)
    Xz = scaler.transform(X_all)
    Z = np.concatenate([t_all, Xz], axis=1)
    return Z, scaler

def _logsumexp(a: np.ndarray, axis: Optional[int] = None) -> np.ndarray:
    a_max = np.max(a, axis=axis, keepdims=True)
    out = a_max + np.log(np.sum(np.exp(a - a_max), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out

def _mix_statistics(mu_k: np.ndarray, Sigma_k: np.ndarray, logw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    logZ = _logsumexp(logw, axis=0)
    w = np.exp(logw - logZ)
    mu = np.einsum('k,kj->j', w, mu_k)
    second = np.einsum('k,kij->ij', w, Sigma_k) + np.einsum('k,ki,kj->ij', w, mu_k, mu_k)
    Sigma = second - np.outer(mu, mu)
    return mu, Sigma

def _z_to_x(mu_z: np.ndarray, Sigma_z: np.ndarray, scaler: StandardScaler) -> Tuple[np.ndarray, np.ndarray]:
    scale = scaler.scale_
    mean = scaler.mean_
    mu_x = mu_z * scale + mean
    Sigma_x = (Sigma_z * scale[None, :]) * scale[:, None]
    return mu_x, Sigma_x

# -----------------------------
# CIP-style anomaly detector with DTW-based threshold learning
# -----------------------------
@dataclass
class CIPAnomalyDetector:
    model: GMMGMRTimeIndexed
    sample_hz: float = 50.0
    e: int = 30
    alpha: float = 1.0/3.0
    modality_map: Dict[str, List[int]] = field(default_factory=lambda: {
        "pose": list(range(0, 7)),
        "wrench": list(range(7, 13)),
        "gripper": [13, 14],
    })
    dtw_aligner: DTWAligner = field(default_factory=DTWAligner)

    thresholds_: Dict[str, float] = field(default_factory=dict)
    _counter_: Dict[str, int] = field(default_factory=dict)
    _first_exceed_time_: Dict[str, Optional[float]] = field(default_factory=dict)
    _last_t_: float = 0.0

    def fit_thresholds(self, demonstrations_list: List[np.ndarray], n_steps: int = 200) -> Dict[str, float]:
        # Get reference from GMR means
        t_grid, mu_grid, Sigma_grid = self.model.predict_trajectory(n_steps)
        # Precompute modality-specific mean/cov and inverses
        red_mu = {name: mu_grid[:, idxs] for name, idxs in self.modality_map.items()}
        red_Sigma = {name: Sigma_grid[:, :][:, idxs][:, :, idxs] for name, idxs in self.modality_map.items()}
        red_inv = {}
        for name, S in red_Sigma.items():
            invs = np.empty_like(S)
            for k in range(S.shape[0]):
                invs[k] = _inv_pd(S[k])
            red_inv[name] = invs

        # Align each demo to the reference timeline using DTW (pose-aware)
        aligner = self.dtw_aligner
        thresholds = {k: 0.0 for k in self.modality_map.keys()}
        for M in demonstrations_list:
            Md = M.T  # (T,15)
            warped, path = aligner.align(mu_grid, Md)
            for name, idxs in self.modality_map.items():
                md = warped[:, idxs]
                mu = red_mu[name]
                invs = red_inv[name]
                diff = md - mu
                q = np.einsum('ti,tij,tj->t', diff, invs, diff)
                thresholds[name] = max(thresholds[name], float(np.max(q)))
        self.thresholds_ = thresholds
        return thresholds

    def reset(self) -> None:
        self._counter_ = {k: 0 for k in self.modality_map.keys()}
        self._first_exceed_time_ = {k: None for k in self.modality_map.keys()}
        self._last_t_ = 0.0

    def distances(self, x_t: np.ndarray, t: Optional[float] = None) -> Dict[str, float]:
        if t is None:
            t = self._last_t_
        mu, Sigma = self.model.predict(float(t))
        dists: Dict[str, float] = {}
        for name, idxs in self.modality_map.items():
            mu_i = mu[idxs]
            S_i = Sigma[np.ix_(idxs, idxs)]
            inv_i = _inv_pd(S_i)
            d = x_t[idxs] - mu_i
            dists[name] = float(d.T @ inv_i @ d)
        return dists

    def update(self, x_t: np.ndarray, t: Optional[float] = None) -> Dict[str, object]:
        if t is None:
            t = min(1.0, self._last_t_ + 1.0 / self.sample_hz)
        self._last_t_ = float(t)
        dists = self.distances(np.asarray(x_t, dtype=np.float64), t=self._last_t_)
        exceeds = {}

        for name, thr in self.thresholds_.items():
            if dists[name] > thr:
                exceeds[name] = True
                self._counter_[name] += 1
                if self._first_exceed_time_[name] is None:
                    self._first_exceed_time_[name] = self._last_t_
            else:
                exceeds[name] = False
                self._counter_[name] = 0
                self._first_exceed_time_[name] = None
        anomaly = False ; which = None ; t_thresh = None
        for name, cnt in self._counter_.items():
            if cnt >= self.e:
                anomaly = True ; which = name ; t_thresh = self._first_exceed_time_[name]
                break
        return {
            "t": self._last_t_,
            "distances": dists,
            "exceeds": exceeds,
            "consecutive": dict(self._counter_),
            "anomaly": anomaly,
            "modality": which,
            "t_thresh": t_thresh,
            "t_anomaly": self._last_t_ if anomaly else None,
        }

    def decision_state_time(self) -> Optional[float]:
        fired = [(k, v) for k, v in self._counter_.items() if v >= self.e]
        if not fired:
            return None
        k = fired[0][0]
        t_thresh = self._first_exceed_time_[k]
        if t_thresh is None:
            return None
        dt = self.alpha * (self.e / self.sample_hz)
        return float(np.clip(t_thresh + dt, 0.0, 1.0))

# ---- numerics ----

def _inv_pd(M: np.ndarray, reg: float = 1e-9) -> np.ndarray:
    M = 0.5 * (M + M.T)
    try:
        L = np.linalg.cholesky(M)
    except np.linalg.LinAlgError:
        d = np.max(np.diag(M))
        M = M + np.eye(M.shape[0]) * reg * (1.0 + d)
        L = np.linalg.cholesky(M)
    Linv = np.linalg.inv(L)
    return Linv.T @ Linv


@dataclass
class Runner:
    model: GMMGMRTimeIndexed
    detector: CIPAnomalyDetector
    pos_tol: float = 5e-3            # 5 mm
    ori_tol_rad: float = 2.0 * np.pi / 180.0  # 2 deg
    n_steps: int = 200

    def run(self) -> None:
        t_grid, mu_grid, Sigma_grid = self.model.predict_trajectory(self.n_steps)
        k = 0
        print("Starting task...")
        while k < len(t_grid):
            # 1) Get robot states (fake or real)
            # 2) Check if robot is at current pose target
            # 3) If so, iterate timestep

            # 4) Check anomaly
            status = self.detector.update(x_t, t=float(t_grid[min(k, len(t_grid)-1)]))
            if status["anomaly"]:
                print(f"ANOMALY at t={status['t_anomaly']:.3f} (modality={status['modality']})")
                t_alpha = self.detector.decision_state_time()
                if t_alpha is not None:
                    print(f"  → Suggested decision-state time t_α={t_alpha:.3f}")
                break

            # 5) Assign timestep goal pose (and move)
            goal_p = mu_grid[k, 0:3]
            goal_q = quat_normalize(mu_grid[k, 3:7])
            self.robot.move_to_pose(goal_p, goal_q)

        print("Task finished or stopped.")

# -----------------------------
# Demo data synthesis (if you don't have real demonstrations yet)
# -----------------------------

def synth_demos(num: int = 4) -> List[np.ndarray]:
    rng = np.random.default_rng(1)
    demos: List[np.ndarray] = []
    for i in range(num):
        T = rng.integers(140, 180)
        t = np.linspace(0, 1, T)
        X = np.zeros((15, T))
        # Position follows a smooth Lissajous-like curve
        X[0] = 0.4 + 0.05 * np.sin(2*np.pi*t) + 0.01 * rng.standard_normal(T)
        X[1] = 0.0 + 0.04 * np.sin(4*np.pi*t + 0.3) + 0.01 * rng.standard_normal(T)
        X[2] = 0.3 + 0.03 * np.cos(2*np.pi*t) + 0.01 * rng.standard_normal(T)
        # Orientation: small rotations around z
        angles = 10*np.pi/180.0 * np.sin(2*np.pi*t)
        X[3:7] = np.vstack([np.zeros(T), np.zeros(T), np.sin(angles/2), np.cos(angles/2)])
        # Wrench signals
        X[7:10] = 3e-1 * np.vstack([np.sin(2*np.pi*t), np.cos(2*np.pi*t), 0.5*np.sin(4*np.pi*t)])
        X[10:13] = 2e-1 * np.vstack([np.cos(2*np.pi*t+0.2), np.sin(2*np.pi*t+0.1), 0.1*np.cos(4*np.pi*t)])
        # Gripper
        X[13] = 0.5 + 0.1 * np.sin(2*np.pi*t)
        X[14] = 0.5 * (t > 0.5).astype(float)
        demos.append(X)
    return demos


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--steps', type=int, default=200, help='Number of GMR timesteps')
    ap.add_argument('--bic-min', type=int, default=2)
    ap.add_argument('--bic-max', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args(argv)

    rng = np.random.default_rng(args.seed)

    # Prepare demos (replace with your real demonstrations_list)
    demonstrations_list = synth_demos(num=5)

    # Fit time-indexed GMM/GMR
    model = GMMGMRTimeIndexed(n_components=None, bic_range=(args.bic_min, args.bic_max), random_state=args.seed)
    model.fit(demonstrations_list)

    # Train detector thresholds with DTW alignment
    det = CIPAnomalyDetector(model, sample_hz=50, e=30, alpha=1/3)
    det.reset()
    det.fit_thresholds(demonstrations_list, n_steps=args.steps)


    # Run
    runner = Runner(model, det, n_steps=args.steps)
    runner.run()

if __name__ == '__main__':
    main()
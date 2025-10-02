
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

POSE_IDXS = list(range(0, 7))  # p(0..2), q(3..6)

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
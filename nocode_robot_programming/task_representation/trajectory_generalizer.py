from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import math
import numpy as np

from numpy.typing import ArrayLike
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


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

if __name__ == "__main__":
    
    
    model = GMMGMRTimeIndexed(n_components=None, bic_range=(2, 12), random_state=0)
    
    model.fit(demonstrations_list)


    t_grid, mu_grid, Sigma_grid = model.predict_trajectory(n_steps)
    mu, Sigma = model.predict(float(t))

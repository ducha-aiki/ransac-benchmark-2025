import numba as nb
import numpy as np

from skimage.measure import ransac as skransac
from skimage.transform import FundamentalMatrixTransform

import numpy as np

# --------------------------
# NumPy helpers (from before)
# --------------------------

def convert_points_to_homogeneous_np(points: np.ndarray) -> np.ndarray:
    B, N, _ = points.shape
    ones = np.ones((B, N, 1), dtype=points.dtype)
    return np.concatenate([points, ones], axis=-1)

def convert_points_from_homogeneous_np(points_h: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    w = points_h[..., 2:3]
    w_safe = np.where(np.abs(w) > eps, w, np.sign(w) * eps + (w == 0) * eps)
    return points_h[..., :2] / w_safe

def transform_points_np(H: np.ndarray, points: np.ndarray) -> np.ndarray:
    pts_h = convert_points_to_homogeneous_np(points)             # (B,N,3)
    pts_h_t = np.einsum('bij,bnj->bni', H, pts_h)                # (B,N,3)
    return convert_points_from_homogeneous_np(pts_h_t)

def normalize_points_np(points: np.ndarray, eps: float = 1e-8):
    """Isotropic (Hartley) normalization. points: (B,N,2). Returns (points_norm, T)."""
    assert points.ndim == 3 and points.shape[-1] == 2
    B, N, _ = points.shape
    x_mean = points.mean(axis=1, keepdims=True)                  # (B,1,2)
    centered = points - x_mean                                   # (B,N,2)
    scale = np.linalg.norm(centered, axis=-1).mean(axis=1)       # (B,)
    scale = np.sqrt(2.0) / (scale + eps)
    T = np.zeros((B, 3, 3), dtype=points.dtype)
    T[:, 0, 0] = scale
    T[:, 1, 1] = scale
    T[:, 0, 2] = -scale * x_mean[..., 0, 0]
    T[:, 1, 2] = -scale * x_mean[..., 0, 1]
    T[:, 2, 2] = 1.0
    pts_n = transform_points_np(T, points)
    return pts_n, T

def normalize_transformation_np(M: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Scale matrices by their bottom-right entry (if |.|>eps)."""
    norm_val = M[..., -1:, -1:]           # (...,1,1)
    cond = np.abs(norm_val) > eps
    denom = norm_val + eps
    return np.where(cond, M / denom, M)



def _svd_last_two_v_np(X: np.ndarray):
    B = X.shape[0]
    f1 = np.zeros((B, 3, 3), dtype=X.dtype)
    f2 = np.zeros((B, 3, 3), dtype=X.dtype)
    for b in range(B):
        _, _, Vt = np.linalg.svd(X[b], full_matrices=False)  # X[b]: (7,9)
        f1[b] = Vt[-2].reshape(3, 3)
        f2[b] = Vt[-1].reshape(3, 3)
    return f1, f2

def _trace_batch(A: np.ndarray) -> np.ndarray:
    return A[:, 0, 0] + A[:, 1, 1] + A[:, 2, 2]

def _solve_cubic_real_roots_np(coeffs: np.ndarray, tol_imag: float = 1e-10) -> np.ndarray:
    B = coeffs.shape[0]
    roots_out = np.zeros((B, 3), dtype=coeffs.dtype)
    for b in range(B):
        a, b1, c, d = coeffs[b]
        poly = np.array([a, b1, c, d], dtype=coeffs.dtype) if not np.isclose(a, 0) else np.array([b1, c, d], dtype=coeffs.dtype)
        r = np.roots(poly)
        real = r[np.abs(r.imag) < tol_imag].real
        real = np.sort(real)
        roots_out[b, :min(3, real.size)] = real[:3]
    return roots_out

def _poly_coeffs_by_sampling(f1: np.ndarray, f2: np.ndarray) -> np.ndarray:
    """
    Stable cubic coefficients for g(λ) = det( f2 + λ*(f1 - f2) ) = a3 λ^3 + a2 λ^2 + a1 λ + a0
    Compute g at λ ∈ {0, 1, -1, 2} and solve the 4x4 linear system for [a3,a2,a1,a0] per batch.
    Inputs:
        f1, f2: (B,3,3)
    Returns:
        coeffs: (B,4) with [a3, a2, a1, a0]
    """
    B = f1.shape[0]
    D = f1 - f2
    # sample g(λ) at λ = 0, 1, -1, 2
    g0  = np.linalg.det(f2)                  # g(0)  = a0
    g1  = np.linalg.det(f2 + D)              # g(1)  = a3 + a2 + a1 + a0
    g_1 = np.linalg.det(f2 - D)              # g(-1) = -a3 + a2 - a1 + a0
    g2  = np.linalg.det(f2 + 2.0 * D)        # g(2)  = 8a3 + 4a2 + 2a1 + a0

    # Build and invert the same 4x4 system once (constant)
    # [ λ^3 λ^2 λ 1 ] rows for λ ∈ {0,1,-1,2}
    M = np.array([
        [0.0**3, 0.0**2, 0.0, 1.0],
        [1.0**3, 1.0**2, 1.0, 1.0],
        [(-1.0)**3, (-1.0)**2, -1.0, 1.0],
        [2.0**3, 2.0**2, 2.0, 1.0],
    ], dtype=np.float64)   # shape (4,4)
    Minv = np.linalg.inv(M)

    G = np.stack([g0, g1, g_1, g2], axis=1)  # (B,4)
    # coeffs = [a3, a2, a1, a0] per batch
    coeffs = (Minv @ G.T).T                  # (B,4)
    return coeffs

def run_7point_np(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    NumPy 7-point with OpenCV-style stable cubic coeffs and post-unnorm rank-2 enforcement.
    points1, points2: (B,7,2) -> (B,3,3,3)
    """
    assert points1.shape == points2.shape and points1.shape[1:] == (7, 2)
    p1 = np.asarray(points1, dtype=np.float64)
    p2 = np.asarray(points2, dtype=np.float64)
    B = p1.shape[0]

    # Normalize points
    p1n, T1 = normalize_points_np(p1)
    p2n, T2 = normalize_points_np(p2)

    x1, y1 = p1n[..., 0:1], p1n[..., 1:2]
    x2, y2 = p2n[..., 0:1], p2n[..., 1:2]
    ones = np.ones_like(x1)

    # Build 7x9 system per batch
    X = np.concatenate([x2 * x1, x2 * y1, x2,
                        y2 * x1, y2 * y1, y2,
                        x1, y1, ones], axis=-1)  # (B,7,9)

    # Nullspace basis
    f1 = np.empty((B, 3, 3), dtype=np.float64)
    f2 = np.empty((B, 3, 3), dtype=np.float64)
    for b in range(B):
        _, _, Vt = np.linalg.svd(X[b], full_matrices=False)
        f1[b] = Vt[-2].reshape(3, 3)
        f2[b] = Vt[-1].reshape(3, 3)

    # Stable cubic coefficients by determinant sampling (no inverses)
    coeffs = _poly_coeffs_by_sampling(f1, f2)  # (B,4) = [a3,a2,a1,a0]

    # Solve cubic per batch, keep up to 3 real roots
    roots = np.zeros((B, 3), dtype=np.float64)
    for b in range(B):
        a3, a2, a1, a0 = coeffs[b]
        r = np.roots([a3, a2, a1, a0])
        real = np.sort(r[np.abs(r.imag) < 1e-10].real)
        roots[b, :min(3, real.size)] = real[:3]

    # Compose candidate Fs in normalized domain: F(λ) = λ f1 + (1-λ) f2
    F_norm = np.empty((B, 3, 3, 3), dtype=np.float64)
    for b in range(B):
        for k in range(3):
            lam = roots[b, k]
            F_norm[b, k] = lam * f1[b] + (1.0 - lam) * f2[b]

    # Unnormalize: F' = T2^T @ F @ T1
    G = np.einsum('bkij,bjm->bkim', F_norm, T1)                   # F @ T1
    F_unn = np.einsum('bij,bkjm->bkim', T2.transpose(0, 2, 1), G) # T2^T @ (..)

    # Rank-2 enforcement per candidate (post-unnormalization)
    for b in range(B):
        for k in range(3):
            U, S, Vt = np.linalg.svd(F_unn[b, k])
            S[-1] = 0.0
            F_unn[b, k] = U @ np.diag(S) @ Vt

    # Final scale normalization
    F_unn = normalize_transformation_np(F_unn).astype(points1.dtype)
    return F_unn



# --------------------------------
# skimage-style transform wrapper
# --------------------------------

class FundamentalMatrixTransform7pt:
    """Fundamental matrix transformation with 7- or 8-point estimation.

    - If exactly 7 correspondences are given, uses NumPy 7-point (up to 3 solutions),
      selecting the one that minimizes mean Sampson distance.
    - If 8+ correspondences are given, falls back to classic normalized 8-point.

    API matches skimage.transform.FundamentalMatrixTransform for estimate(), residuals(), etc.
    """

    def __init__(self, matrix=None, *, dimensionality=2):
        if matrix is None:
            matrix = np.eye(dimensionality + 1)
        else:
            matrix = np.asarray(matrix)
            dimensionality = matrix.shape[0] - 1
            if matrix.shape != (dimensionality + 1, dimensionality + 1):
                raise ValueError("Invalid shape of transformation matrix")
        self.params = matrix
        if dimensionality != 2:
            raise NotImplementedError(
                f'{self.__class__} is only implemented for 2D coordinates '
                '(i.e. 3D transformation matrices).'
            )

    def __call__(self, coords):
        coords = np.asarray(coords)
        coords_h = np.column_stack([coords, np.ones(coords.shape[0])])
        return coords_h @ self.params.T

    @property
    def inverse(self):
        return type(self)(matrix=self.params.T)

    # --- Normalization just like skimage's _center_and_normalize_points, but here we reuse our helpers.
    def _normalize_points_single(self, pts: np.ndarray):
        """Return (T, pts_norm) for (N,2) pts."""
        pts_b = pts[None, ...]  # (1,N,2)
        pts_n, T = normalize_points_np(pts_b)
        return T[0], pts_n[0]

    def _setup_constraint_matrix_8point(self, src, dst):
        """Return (F_normalized, src_T, dst_T) for 8-point flow."""
        src = np.asarray(src); dst = np.asarray(dst)
        if src.shape != dst.shape:
            raise ValueError('src and dst shapes must be identical.')
        if src.shape[0] < 8:
            raise ValueError('src.shape[0] must be equal or larger than 8.')

        try:
            src_T, src_n = self._normalize_points_single(src)
            dst_T, dst_n = self._normalize_points_single(dst)
        except ZeroDivisionError:
            nan3 = np.full((3, 3), np.nan)
            self.params = nan3
            return nan3, nan3, nan3

        # Build A for dst' * F * src = 0
        N = src_n.shape[0]
        A = np.ones((N, 9), dtype=src_n.dtype)
        A[:, 0] = src_n[:, 0] * dst_n[:, 0]
        A[:, 1] = src_n[:, 1] * dst_n[:, 0]
        A[:, 2] = dst_n[:, 0]
        A[:, 3] = src_n[:, 0] * dst_n[:, 1]
        A[:, 4] = src_n[:, 1] * dst_n[:, 1]
        A[:, 5] = dst_n[:, 1]
        A[:, 6] = src_n[:, 0]
        A[:, 7] = src_n[:, 1]
        # A[:, 8] already ones

        # Solve nullspace
        _, _, Vt = np.linalg.svd(A)
        F_norm = Vt[-1, :].reshape(3, 3)

        return F_norm, src_T, dst_T

    def estimate(self, src, dst):
        """Estimate F using 7-point (N==7) or 8-point (N>=8). Returns bool."""
        src = np.asarray(src)
        dst = np.asarray(dst)
        if src.shape != dst.shape:
            return False

        # Ensure 2D (N,2) inputs; if something upstream passes (1,N,2), squeeze batch dim.
        if src.ndim == 3 and src.shape[0] == 1:
            src = src[0]
            dst = dst[0]
        if src.ndim != 2 or src.shape[1] != 2:
            return False

        N = src.shape[0]
        if N < 7:
            return False

        # ---------- STRICT 7-point ----------
        if N == 7:
            # Double-check shape before calling the 7-point solver
            assert src.shape == (7, 2) and dst.shape == (7, 2), "7-point requires exactly 7 correspondences"
            try:
                F_cands = run_7point_np(src[None, ...], dst[None, ...])[0]  # (3,3,3)
            except Exception:
                return False

            # Pick the most robust candidate by median Sampson error
            errs = []
            for k in range(3):
                Fk = F_cands[k]
                if not np.all(np.isfinite(Fk)):
                    errs.append(np.inf)
                else:
                    e = self._sampson_errors(Fk, src, dst)
                    errs.append(np.median(e))
            best = int(np.argmin(errs))
            if not np.isfinite(errs[best]):
                return False
            self.params = F_cands[best]
            return True

        # ---------- N >= 8 -> 8-point ----------
        F_norm, src_T, dst_T = self._setup_constraint_matrix_8point(src, dst)
        if F_norm is None:
            return False
        try:
            U, S, Vt = np.linalg.svd(F_norm)
        except np.linalg.LinAlgError:
            return False
        S[2] = 0.0
        F_rank2 = U @ np.diag(S) @ Vt
        self.params = dst_T.T @ F_rank2 @ src_T
        return np.all(np.isfinite(self.params))



    def _sampson_errors(self, F, src, dst):
        """Return Sampson distance per correspondence."""
        src_h = np.column_stack([src, np.ones(src.shape[0])])   # (N,3)
        dst_h = np.column_stack([dst, np.ones(dst.shape[0])])   # (N,3)
        F_src = (F @ src_h.T)                                   # (3,N)
        Ft_dst = (F.T @ dst_h.T)                                # (3,N)
        num = np.sum(dst_h * F_src.T, axis=1)                   # (N,)
        den = F_src[0]**2 + F_src[1]**2 + Ft_dst[0]**2 + Ft_dst[1]**2
        return np.abs(num) / np.sqrt(den + 1e-15)

    def residuals(self, src, dst):
        """Sampson distance (same as skimage)."""
        return self._sampson_errors(self.params, np.asarray(src), np.asarray(dst))

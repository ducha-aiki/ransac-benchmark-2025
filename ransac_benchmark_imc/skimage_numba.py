import numba as nb
import numpy as np

from skimage.measure import ransac as skransac
from skimage.transform import FundamentalMatrixTransform

import numpy as np
from numba import njit

# ----------------------------
# small numeric helpers (numba)
# ----------------------------


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

@njit(cache=True)
def _cbrt(x: float) -> float:
    ax = abs(x)
    return np.sign(x) * (ax ** (1.0 / 3.0))

# Vandermonde for λ ∈ {0, 1, -1, 2}
_VAND = np.array([
    [0.0**3, 0.0**2, 0.0, 1.0],
    [1.0**3, 1.0**2, 1.0, 1.0],
    [(-1.0)**3, (-1.0)**2, -1.0, 1.0],
    [2.0**3, 2.0**2, 2.0, 1.0],
], dtype=np.float64)
_MINV_SAMPLE = np.linalg.inv(_VAND)

@njit(cache=True)
def _solve_cubic_real_cardano(a3: float, a2: float, a1: float, a0: float, roots_out: np.ndarray) -> int:
    """
    Solve a3*x^3 + a2*x^2 + a1*x + a0 = 0 for *real* roots.
    Writes up to 3 real roots into roots_out[0:count] (ascending).
    Returns count.
    """
    # Handle degenerate leading coefficient (fall back to quadratic/linear)
    if abs(a3) < 1e-18:
        # quadratic: a2 x^2 + a1 x + a0 = 0
        if abs(a2) < 1e-18:
            if abs(a1) < 1e-18:
                return 0
            roots_out[0] = -a0 / a1
            return 1
        D = a1 * a1 - 4.0 * a2 * a0
        if D < 0.0:
            return 0
        elif abs(D) < 1e-18:
            roots_out[0] = (-a1) / (2.0 * a2)
            return 1
        else:
            sqrtD = np.sqrt(D)
            r1 = (-a1 - sqrtD) / (2.0 * a2)
            r2 = (-a1 + sqrtD) / (2.0 * a2)
            if r1 <= r2:
                roots_out[0] = r1
                roots_out[1] = r2
            else:
                roots_out[0] = r2
                roots_out[1] = r1
            return 2

    # Normalize to monic: x^3 + b x^2 + c x + d = 0
    b = a2 / a3
    c = a1 / a3
    d = a0 / a3

    # Depressed cubic y^3 + p y + q = 0  with x = y - b/3
    b3 = b / 3.0
    p = c - b * b3
    q = 2.0 * b3 * b3 * b3 - b3 * c + d

    # Discriminant
    D = 0.25 * q * q + (p * p * p) / 27.0

    if D > 1e-18:
        # One real root
        sqrtD = np.sqrt(D)
        u = _cbrt(-0.5 * q + sqrtD)
        v = _cbrt(-0.5 * q - sqrtD)
        y = u + v
        roots_out[0] = y - b3
        return 1
    elif D >= -1e-18:
        # Multiple real roots (D ~ 0)
        u = _cbrt(-0.5 * q)
        y1 = 2.0 * u
        y2 = -u
        r1 = y1 - b3
        r2 = y2 - b3
        # If u is ~0, both roots equal
        if abs(y1 - y2) < 1e-12:
            roots_out[0] = r1
            return 1
        # two real roots; third coincides
        if r1 <= r2:
            roots_out[0] = r1
            roots_out[1] = r2
        else:
            roots_out[0] = r2
            roots_out[1] = r1
        return 2
    else:
        # Three distinct real roots
        # y_k = 2*sqrt(-p/3)*cos((1/3) arccos( (3q)/(2p)*sqrt(-3/p) ) - 2πk/3)
        phi = 0.0
        # guard for p ~ 0
        if abs(p) < 1e-18:
            roots_out[0] = -b3
            return 1
        t = np.sqrt(-4.0 * p / 3.0)  # note: 2*sqrt(-p/3)
        # argument of arccos
        arg = (3.0 * q) / (p * t)
        # clamp for safety
        if arg < -1.0:
            arg = -1.0
        elif arg > 1.0:
            arg = 1.0
        phi = np.arccos(arg) / 3.0

        r0 = 0.5 * t * np.cos(phi) * 2.0 - b3
        r1 = 0.5 * t * np.cos(phi - 2.0 * np.pi / 3.0) * 2.0 - b3
        r2 = 0.5 * t * np.cos(phi - 4.0 * np.pi / 3.0) * 2.0 - b3

        # sort ascending
        if r0 > r1:
            r0, r1 = r1, r0
        if r1 > r2:
            r1, r2 = r2, r1
        if r0 > r1:
            r0, r1 = r1, r0

        roots_out[0] = r0
        roots_out[1] = r1
        roots_out[2] = r2
        return 3

@njit(cache=True)
def _mat33_mul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    C = np.empty((3, 3), dtype=A.dtype)
    for i in range(3):
        for j in range(3):
            s = 0.0
            for k in range(3):
                s += A[i, k] * B[k, j]
            C[i, j] = s
    return C

@njit(cache=True)
def _mat33_T(A: np.ndarray) -> np.ndarray:
    T = np.empty((3, 3), dtype=A.dtype)
    for i in range(3):
        for j in range(3):
            T[i, j] = A[j, i]
    return T

@njit(cache=True)
def _normalize_points_single(pts: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Hartley isotropic normalization for (7,2) pts.
    Returns (pts_norm(7,2), T(3,3)).
    """
    N = pts.shape[0]
    x_mean0 = 0.0
    x_mean1 = 0.0
    for i in range(N):
        x_mean0 += pts[i, 0]
        x_mean1 += pts[i, 1]
    x_mean0 /= N
    x_mean1 /= N

    # mean distance to origin
    md = 0.0
    for i in range(N):
        dx = pts[i, 0] - x_mean0
        dy = pts[i, 1] - x_mean1
        md += np.sqrt(dx * dx + dy * dy)
    md /= N
    scale = np.sqrt(2.0) / (md + 1e-12)

    T = np.zeros((3, 3), dtype=pts.dtype)
    T[0, 0] = scale
    T[1, 1] = scale
    T[0, 2] = -scale * x_mean0
    T[1, 2] = -scale * x_mean1
    T[2, 2] = 1.0

    pts_n = np.empty_like(pts)
    for i in range(N):
        x = pts[i, 0]
        y = pts[i, 1]
        xn = scale * x + T[0, 2]
        yn = scale * y + T[1, 2]
        pts_n[i, 0] = xn
        pts_n[i, 1] = yn
    return pts_n, T

@njit(cache=True)
def _build_A_7(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    """Build 7x9 constraint matrix."""
    A = np.ones((7, 9), dtype=x1.dtype)
    for i in range(7):
        X1 = x1[i]
        Y1 = y1[i]
        X2 = x2[i]
        Y2 = y2[i]
        A[i, 0] = X2 * X1
        A[i, 1] = X2 * Y1
        A[i, 2] = X2
        A[i, 3] = Y2 * X1
        A[i, 4] = Y2 * Y1
        A[i, 5] = Y2
        A[i, 6] = X1
        A[i, 7] = Y1
        # A[i,8] already 1
    return A

from numba import njit

@njit(cache=True)
def _poly_coeffs_sampling_batch(f1: np.ndarray, f2: np.ndarray, coeffs: np.ndarray):
    """
    coeffs[b] <- [a3,a2,a1,a0] using determinant sampling at λ∈{0,1,-1,2}.
    f1,f2: (B,3,3); coeffs: (B,4)
    """
    B = f1.shape[0]
    for b in range(B):
        D = f1[b] - f2[b]
        g0 = np.linalg.det(f2[b])            # λ=0
        g1 = np.linalg.det(f2[b] + D)        # λ=1
        g_1 = np.linalg.det(f2[b] - D)       # λ=-1
        g2 = np.linalg.det(f2[b] + 2.0*D)    # λ=2

        # use the precomputed inverse (read-only global)
        G0 = g0
        G1 = g1
        Gm1 = g_1
        G2 = g2

        # c = _MINV_SAMPLE @ [g0, g1, g_-1, g2]
        c0 = _MINV_SAMPLE[0,0]*G0 + _MINV_SAMPLE[0,1]*G1 + _MINV_SAMPLE[0,2]*Gm1 + _MINV_SAMPLE[0,3]*G2
        c1 = _MINV_SAMPLE[1,0]*G0 + _MINV_SAMPLE[1,1]*G1 + _MINV_SAMPLE[1,2]*Gm1 + _MINV_SAMPLE[1,3]*G2
        c2 = _MINV_SAMPLE[2,0]*G0 + _MINV_SAMPLE[2,1]*G1 + _MINV_SAMPLE[2,2]*Gm1 + _MINV_SAMPLE[2,3]*G2
        c3 = _MINV_SAMPLE[3,0]*G0 + _MINV_SAMPLE[3,1]*G1 + _MINV_SAMPLE[3,2]*Gm1 + _MINV_SAMPLE[3,3]*G2

        coeffs[b, 0] = c0
        coeffs[b, 1] = c1
        coeffs[b, 2] = c2
        coeffs[b, 3] = c3


@njit(cache=True)
def _rank2_project(F: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(F)   # <-- no kwargs
    S[2] = 0.0
    # build diag manually (Numba-friendly)
    D = np.zeros((3,3), dtype=F.dtype)
    D[0,0] = S[0]; D[1,1] = S[1]
    return U @ D @ Vt


@njit(cache=True)
def _normalize_by_bottom_right(F: np.ndarray) -> np.ndarray:
    """Divide F by F[2,2] (if safe) to fix scale."""
    denom = F[2, 2]
    if abs(denom) > 1e-8:
        return F / (denom + 1e-8)
    return F

# ----------------------------
# main: numba 7-point (batched)
# ----------------------------

@njit(cache=True)
def run_7point_numba(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    B = points1.shape[0]
    out = np.zeros((B, 3, 3, 3), dtype=np.float64)

    for b in range(B):
        p1b = np.ascontiguousarray(points1[b].astype(np.float64))
        p2b = np.ascontiguousarray(points2[b].astype(np.float64))

        p1n, T1 = _normalize_points_single(p1b)
        p2n, T2 = _normalize_points_single(p2b)

        x1 = p1n[:, 0]; y1 = p1n[:, 1]
        x2 = p2n[:, 0]; y2 = p2n[:, 1]

        A = _build_A_7(x1, y1, x2, y2)  # (7,9)

        # --- SVD without kwargs; use computed indices (Numba-friendly)
        U, S, Vt = np.linalg.svd(A)
        nrows = Vt.shape[0]

        # build F1 from the penultimate row of Vt without reshape()
        row1 = Vt[nrows - 2]
        F1 = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            base = 3 * i
            F1[i, 0] = row1[base + 0]
            F1[i, 1] = row1[base + 1]
            F1[i, 2] = row1[base + 2]

        # build F2 from the last row of Vt without reshape()
        row2 = Vt[nrows - 1]
        F2 = np.empty((3, 3), dtype=np.float64)
        for i in range(3):
            base = 3 * i
            F2[i, 0] = row2[base + 0]
            F2[i, 1] = row2[base + 1]
            F2[i, 2] = row2[base + 2]

        coeffs = np.zeros((1, 4), dtype=np.float64)
        _poly_coeffs_sampling_batch(F1.reshape(1,3,3), F2.reshape(1,3,3), coeffs)
        a3, a2, a1, a0 = coeffs[0, 0], coeffs[0, 1], coeffs[0, 2], coeffs[0, 3]

        roots = np.zeros(3, dtype=np.float64)
        n_roots = _solve_cubic_real_cardano(a3, a2, a1, a0, roots)

        for k in range(3):
            lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
            Fn = lam * F1 + (1.0 - lam) * F2
            G = _mat33_mul(Fn, T1)
            Funn = _mat33_mul(_mat33_T(T2), G)
            Fproj = _rank2_project(Funn)
            # normalize by bottom-right
            denom = Fproj[2, 2]
            if abs(denom) > 1e-8:
                Fproj = Fproj / (denom + 1e-8)
            out[b, k] = Fproj

    return out.astype(points1.dtype)




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
            if True:#try:
                F_cands = run_7point_numba(src[None, ...], dst[None, ...])[0]  # (3,3,3)
            else:#except Exception:
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

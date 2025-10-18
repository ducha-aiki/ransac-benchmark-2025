import numpy as np
from numba import njit

# =========================
# Small linear-algebra bits
# =========================

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
def _rank2_project(F: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    D = np.zeros((3, 3), dtype=F.dtype)
    D[0, 0] = S[0]; D[1, 1] = S[1]
    return _mat33_mul(_mat33_mul(U, D), Vt)

# =========================
# Normalization (Hartley)
# =========================
@njit(cache=True)
def run_8point_numba_reuse_dynamic(src: np.ndarray, dst: np.ndarray,
                                        A_buf: np.ndarray, G_buf: np.ndarray) -> np.ndarray:
    src_n, T1 = _normalize_points_single(src)
    dst_n, T2 = _normalize_points_single(dst)
    M = src_n.shape[0]

    # Use provided A_buf if it fits; else allocate temp (slow path, but rare)
    if M <= A_buf.shape[0]:
        A = A_buf
    else:
        A = np.empty((M, 9), dtype=np.float64)

    _fill_A_rows(A, src_n, dst_n, M)

    # Build Gram in G_buf (always 9x9)
    for i in range(9):
        for j in range(9):
            s = 0.0
            for r in range(M):
                s += A[r, i] * A[r, j]
            G_buf[i, j] = s

    w, V = np.linalg.eigh(G_buf)
    v = V[:, 0]

    F = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        b = 3*i
        F[i,0]=v[b+0]; F[i,1]=v[b+1]; F[i,2]=v[b+2]
    F = _mat33_mul(_mat33_T(T2), _mat33_mul(F, T1))
    return F


@njit(cache=True)
def _normalize_points_single(pts: np.ndarray):
    """
    Hartley isotropic normalization with RMS distance.
    pts: (N,2) -> returns (pts_norm, T) with
      scale = sqrt(2) / sqrt(mean(dx^2 + dy^2))
    """
    N = pts.shape[0]
    # Welford for mean + sum of squared distances to mean
    mean0 = 0.0; mean1 = 0.0
    m0 = 0.0; m1 = 0.0  # running means
    ss = 0.0
    for i in range(N):
        x = pts[i,0]; y = pts[i,1]
        i1 = i + 1.0
        dx0 = x - m0; dx1 = y - m1
        m0 += dx0 / i1; m1 += dx1 / i1
        mean0 = m0; mean1 = m1
        # use updated mean to accumulate squared distance
        ss += dx0*(x - m0) + dx1*(y - m1)
    # ss is sum of squared distances to mean
    mean_sq = (ss / N)
    if mean_sq <= 1e-24: mean_sq = 1e-24
    scale = np.sqrt(2.0) / np.sqrt(mean_sq)

    T = np.zeros((3,3), dtype=pts.dtype)
    T[0,0]=scale; T[1,1]=scale; T[2,2]=1.0
    T[0,2]=-scale*mean0; T[1,2]=-scale*mean1

    pts_n = np.empty_like(pts)
    tx = T[0,2]; ty = T[1,2]
    for i in range(N):
        pts_n[i,0] = scale*pts[i,0] + tx
        pts_n[i,1] = scale*pts[i,1] + ty
    return pts_n, T


# =========================
# Seven-point: cubic solver
# =========================

@njit(cache=True)
def _stable_quadratic_real(a: float, b: float, c: float, roots_out: np.ndarray) -> int:
    if abs(a) < 1e-18:
        if abs(b) < 1e-18:
            return 0
        roots_out[0] = -c / b
        return 1
    D = b * b - 4.0 * a * c
    if D < 0.0:
        return 0
    if abs(D) < 1e-18:
        roots_out[0] = -b / (2.0 * a)
        return 1
    sqrtD = np.sqrt(D)
    q = -0.5 * (b + (sqrtD if b >= 0.0 else -sqrtD))
    r1 = q / a
    r2 = c / q
    if r1 <= r2:
        roots_out[0] = r1; roots_out[1] = r2
    else:
        roots_out[0] = r2; roots_out[1] = r1
    return 2

@njit(cache=True)
def _solve_cubic_real(a3: float, a2: float, a1: float, a0: float, roots_out: np.ndarray) -> int:
    maxc = max(abs(a3), abs(a2), abs(a1), abs(a0))
    if maxc == 0.0:
        return 0
    s = 1.0 / maxc
    a3 *= s; a2 *= s; a1 *= s; a0 *= s

    if abs(a3) < 1e-18:
        return _stable_quadratic_real(a2, a1, a0, roots_out)

    b = a2 / a3
    c = a1 / a3
    d = a0 / a3

    b3 = b / 3.0
    p = c - b * b3
    q = 2.0 * b3 * b3 * b3 - b3 * c + d

    half_q = 0.5 * q
    D = half_q * half_q + (p * p * p) / 27.0
    epsD = 1e-18

    def _cbrt_loc(x):
        ax = abs(x)
        return np.sign(x) * (ax ** (1.0 / 3.0))

    if D > epsD:
        sqrtD = np.sqrt(D)
        u = _cbrt_loc(-half_q + sqrtD)
        v = _cbrt_loc(-half_q - sqrtD)
        roots_out[0] = (u + v) - b3
        return 1

    if D >= -epsD:
        u = _cbrt_loc(-half_q)
        y1 = 2.0 * u
        y2 = -u
        r1 = y1 - b3
        r2 = y2 - b3
        if abs(r1 - r2) < 1e-12:
            roots_out[0] = r1
            return 1
        if r1 <= r2:
            roots_out[0] = r1; roots_out[1] = r2
        else:
            roots_out[0] = r2; roots_out[1] = r1
        return 2

    if abs(p) < 1e-24:
        y = _cbrt_loc(-q)
        roots_out[0] = y - b3
        return 1

    # Three distinct real roots — FIXED AMPLITUDE
    t = 2.0 * np.sqrt(-p / 3.0)
    arg = (3.0 * q) / (2.0 * p) * np.sqrt(-3.0 / p)
    if arg < -1.0: arg = -1.0
    elif arg > 1.0: arg = 1.0
    phi = np.arccos(arg) / 3.0

    r0 =  t * np.cos(phi)               - b3
    r1 =  t * np.cos(phi - 2.0*np.pi/3) - b3
    r2 =  t * np.cos(phi - 4.0*np.pi/3) - b3

    if r0 > r1: r0, r1 = r1, r0
    if r1 > r2: r1, r2 = r2, r1
    if r0 > r1: r0, r1 = r1, r0

    cnt = 0
    last = 0.0
    for k in range(3):
        val = r0 if k == 0 else (r1 if k == 1 else r2)
        if cnt == 0 or abs(val - last) > 1e-12:
            roots_out[cnt] = val
            last = val
            cnt += 1
    return cnt



@njit(cache=True)
def _draw_k_unique(N: int, k: int, rng_state: np.int64, out_idx: np.ndarray) -> np.int64:
    """
    Uniformly sample k unique indices from range [0, N).
    - out_idx must be preallocated with length >= k, dtype int64 (or int32 consistently).
    - Returns the updated rng_state.
    """
    visited = np.zeros(N, dtype=np.uint8)   # O(N) tiny bytes, reset each call
    chosen = 0
    while chosen < k:
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        j = int(rng_state % np.int64(N))
        if visited[j] == 0:
            visited[j] = 1
            out_idx[chosen] = j
            chosen += 1
    return rng_state


@njit(cache=True)
def _seven_point_coeffs_explicit(f1: np.ndarray, f2: np.ndarray, coeffs_out: np.ndarray) -> int:
    """
    Compute cubic coefficients for det(lambda * f1 + f2) == 0
    following COLMAP's explicit formula.
    Inputs:
      f1, f2: (9,) row-major (f00,f01,f02,f10,f11,f12,f20,f21,f22)
              with f1 already replaced by (v1 - v2) so that F = lambda*f1 + f2
      coeffs_out: (4,) will be set to [a3, a2, a1, a0] of monic cubic
                  i.e., a3=1, a2=coeffs[1]/coeffs[0], a1=coeffs[2]/coeffs[0], a0=coeffs[3]/coeffs[0]
    Returns:
      1 if OK, 0 if degenerate (leading coeff ~ 0)
    """
    # shorthand
    a0,a1,a2,a3,a4,a5,a6,a7,a8 = f1[0],f1[1],f1[2],f1[3],f1[4],f1[5],f1[6],f1[7],f1[8]
    b0,b1,b2,b3,b4,b5,b6,b7,b8 = f2[0],f2[1],f2[2],f2[3],f2[4],f2[5],f2[6],f2[7],f2[8]

    # t0..t5 as in COLMAP
    t0 = a4 * a8 - a5 * a7
    t1 = a3 * a8 - a5 * a6
    t2 = a3 * a7 - a4 * a6
    t3 = b4 * b8 - b5 * b7
    t4 = b3 * b8 - b5 * b6
    t5 = b3 * b7 - b4 * b6

    # coeffs[0..3] before making monic
    c0 = a0 * t0 - a1 * t1 + a2 * t2
    if abs(c0) < 1e-16:
        return 0

    c1 = (b0 * t0 - b1 * t1 + b2 * t2
          - b3 * (a1 * a8 - a2 * a7)
          + b4 * (a0 * a8 - a2 * a6)
          - b5 * (a0 * a7 - a1 * a6)
          + b6 * (a1 * a5 - a2 * a4)
          - b7 * (a0 * a5 - a2 * a3)
          + b8 * (a0 * a4 - a1 * a3))

    c2 = (a0 * t3 - a1 * t4 + a2 * t5
          - a3 * (b1 * b8 - b2 * b7)
          + a4 * (b0 * b8 - b2 * b6)
          - a5 * (b0 * b7 - b1 * b6)
          + a6 * (b1 * b5 - b2 * b4)
          - a7 * (b0 * b5 - b2 * b3)
          + a8 * (b0 * b4 - b1 * b3))

    c3 = b0 * t3 - b1 * t4 + b2 * t5

    # make monic: divide tail by c0
    inv = 1.0 / c0
    coeffs_out[0] = 1.0
    coeffs_out[1] = c1 * inv
    coeffs_out[2] = c2 * inv
    coeffs_out[3] = c3 * inv
    return 1


@njit(cache=True)
def _build_A_7(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    A = np.ones((7, 9), dtype=x1.dtype)
    for i in range(7):
        X1, Y1, X2, Y2 = x1[i], y1[i], x2[i], y2[i]
        A[i, 0] = X2 * X1; A[i, 1] = X2 * Y1; A[i, 2] = X2
        A[i, 3] = Y2 * X1; A[i, 4] = Y2 * Y1; A[i, 5] = Y2
        A[i, 6] = X1;      A[i, 7] = Y1
    return A

# =========================
# Seven- & Eight-point solvers
# =========================

@njit(cache=True)
def run_7point_numba(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Batched 7-point: (B,7,2) -> (B,3,3,3) candidates.
    """
    B = points1.shape[0]
    out = np.zeros((3, 3, 3), dtype=np.float64)


    p1n, T1 = _normalize_points_single(points1.astype(np.float64))
    p2n, T2 = _normalize_points_single(points2.astype(np.float64))

    x1, y1 = p1n[:, 0], p1n[:, 1]
    x2, y2 = p2n[:, 0], p2n[:, 1]
    A = _build_A_7(x1, y1, x2, y2)

    U, S, Vt = np.linalg.svd(A)
    row1 = Vt[-2]; row2 = Vt[-1]

    F1 = np.empty((3, 3), dtype=np.float64)
    F2 = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        base = 3 * i
        F1[i, 0] = row1[base + 0]; F1[i, 1] = row1[base + 1]; F1[i, 2] = row1[base + 2]
        F2[i, 0] = row2[base + 0]; F2[i, 1] = row2[base + 1]; F2[i, 2] = row2[base + 2]


    # Prepare f1,f2 as 9-vectors in row-major and reparameterize: f1 = v1 - v2
    f1v = np.empty(9, dtype=np.float64)
    f2v = np.empty(9, dtype=np.float64)
    for i in range(3):
        base = 3 * i
        f1v[base+0] = row1[base+0] - row2[base+0]
        f1v[base+1] = row1[base+1] - row2[base+1]
        f1v[base+2] = row1[base+2] - row2[base+2]
        f2v[base+0] = row2[base+0]
        f2v[base+1] = row2[base+1]
        f2v[base+2] = row2[base+2]

    coeffs = np.empty(4, dtype=np.float64)
    ok = _seven_point_coeffs_explicit(f1v, f2v, coeffs)
    if ok == 0:
        # degenerate: no valid cubic; produce a safe fallback (all zeros)
        for k in range(3):
            out[k, :, :] = 0.0

    # coeffs = [1, a2, a1, a0] for monic cubic; solve for real roots
    a3, a2c, a1c, a0c = coeffs[0], coeffs[1], coeffs[2], coeffs[3]

    roots = np.zeros(3, dtype=np.float64)
    n_roots = _solve_cubic_real(a3, a2c, a1c, a0c, roots)

    for k in range(3):
        lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
        # F(λ) = λ*v1 + (1-λ)*v2  == λ*F1 + (1-λ)*F2  (your existing assembly)
        Fn = lam * F1 + (1.0 - lam) * F2
        G = _mat33_mul(Fn, T1)
        F = _mat33_mul(_mat33_T(T2), G)
        F = _rank2_project(F)
        d = F[2, 2]
        if abs(d) > 1e-8:
            F = F / (d + 1e-8)
        out[k] = F
    return out


@njit(cache=True)
def run_7point_numba_reuse(src: np.ndarray, dst: np.ndarray,
                                A_buf: np.ndarray, G_buf: np.ndarray) -> np.ndarray:
    out = np.zeros((3,3,3), dtype=np.float64)
    src_n, T1 = _normalize_points_single(src.astype(np.float64))
    dst_n, T2 = _normalize_points_single(dst.astype(np.float64))
    _fill_A_rows(A_buf, src_n, dst_n, 7)
    w, V = _eig_smallest_vectors_AtA(A_buf, 7, G_buf)
    row1 = V[:, 0]  # smallest
    row2 = V[:, 1]  # second smallest

    # Build F1,F2
    F1 = np.empty((3,3), dtype=np.float64)
    F2 = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        b = 3*i
        F1[i,0]=row1[b+0]; F1[i,1]=row1[b+1]; F1[i,2]=row1[b+2]
        F2[i,0]=row2[b+0]; F2[i,1]=row2[b+1]; F2[i,2]=row2[b+2]

    # cubic on f1=v1-v2, f2=v2
    f1v = np.empty(9, dtype=np.float64); f2v = np.empty(9, dtype=np.float64)
    for i in range(3):
        b = 3*i
        f1v[b+0]=row1[b+0]-row2[b+0]; f1v[b+1]=row1[b+1]-row2[b+1]; f1v[b+2]=row1[b+2]-row2[b+2]
        f2v[b+0]=row2[b+0];           f2v[b+1]=row2[b+1];           f2v[b+2]=row2[b+2]

    coeffs = np.empty(4, dtype=np.float64)
    ok = _seven_point_coeffs_explicit(f1v, f2v, coeffs)
    if ok == 0: return out

    roots = np.zeros(3, dtype=np.float64)
    n_roots = _solve_cubic_real(coeffs[0], coeffs[1], coeffs[2], coeffs[3], roots)

    for k in range(3):
        lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
        Fn = lam * F1 + (1.0 - lam) * F2
        F = _mat33_mul(_mat33_T(T2), _mat33_mul(Fn, T1))
        out[k] = F
    return out


@njit(cache=True)
def run_7point_numba_reuse_old(src: np.ndarray, dst: np.ndarray,
                           A_buf: np.ndarray, G_buf: np.ndarray) -> np.ndarray:
    """
    src,dst: (7,2)
    A_buf  : (>=7, 9) reused workspace
    G_buf  : (9,9) reused Gram matrix buffer
    Returns: (3,3,3) candidate Fs (no rank-2 projection here; do it after selection)
    """
    out = np.zeros((3,3,3), dtype=np.float64)

    src_n, T1 = _normalize_points_single(src.astype(np.float64))
    dst_n, T2 = _normalize_points_single(dst.astype(np.float64))

    # Fill only first 7 rows
    _fill_A_rows(A_buf, src_n, dst_n, 7)

    # --- Option A: use SVD (simple, a bit slower)
    # U, S, Vt = np.linalg.svd(A_buf[:7, :])
    # row1 = Vt[-2]; row2 = Vt[-1]

    # --- Option B: build G = AᵀA and use eigh (fast & alloc-free)
    _gram_AtA(A_buf, 7, G_buf)
    w, V = np.linalg.eigh(G_buf)  # ascending eigenvalues
    row1 = V[:, 0]                 # smallest
    row2 = V[:, 1]                 # 2nd smallest

    # Make F1,F2 (3x3)
    F1 = np.empty((3,3), dtype=np.float64)
    F2 = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        base = 3*i
        F1[i,0]=row1[base+0]; F1[i,1]=row1[base+1]; F1[i,2]=row1[base+2]
        F2[i,0]=row2[base+0]; F2[i,1]=row2[base+1]; F2[i,2]=row2[base+2]

    # Prepare cubic
    f1v = np.empty(9, dtype=np.float64)
    f2v = np.empty(9, dtype=np.float64)
    for i in range(3):
        base = 3*i
        f1v[base+0] = row1[base+0] - row2[base+0]
        f1v[base+1] = row1[base+1] - row2[base+1]
        f1v[base+2] = row1[base+2] - row2[base+2]
        f2v[base+0] = row2[base+0]
        f2v[base+1] = row2[base+1]
        f2v[base+2] = row2[base+2]

    coeffs = np.empty(4, dtype=np.float64)
    ok = _seven_point_coeffs_explicit(f1v, f2v, coeffs)
    if ok == 0:
        return out  # zeros

    roots = np.zeros(3, dtype=np.float64)
    n_roots = _solve_cubic_real(coeffs[0], coeffs[1], coeffs[2], coeffs[3], roots)

    # Build candidates (defer rank-2 projection & scaling)
    for k in range(3):
        lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
        Fn = lam * F1 + (1.0 - lam) * F2
        F = _mat33_mul(_mat33_T(T2), _mat33_mul(Fn, T1))
        out[k] = F
    return out


### RFC check ###
@njit(cache=True)
def _rfc_check(F: np.ndarray) -> bool:
    # Use float64 math
    F00, F01, F02 = F[0,0], F[0,1], F[0,2]
    F10, F11, F12 = F[1,0], F[1,1], F[1,2]
    F20, F21, F22 = F[2,0], F[2,1], F[2,2]

    den = (F00*F01*F20*F22 - F00*F02*F20*F21 +
           F01*F01*F21*F22 - F01*F02*F21*F21 +
           F10*F11*F20*F22 - F10*F12*F20*F21 +
           F11*F11*F21*F22 - F11*F12*F21*F21)

    num = -F22 * (F01*F02*F22 - F02*F02*F21 + F11*F12*F22 - F12*F12*F21)
    if num * den < 0.0:
        return False

    den = (F00*F10*F02*F22 - F00*F20*F02*F12 +
           F10*F10*F12*F22 - F10*F20*F12*F12 +
           F01*F11*F02*F22 - F01*F21*F02*F12 +
           F11*F11*F12*F22 - F11*F21*F12*F12)

    num = -F22 * (F10*F20*F22 - F20*F20*F12 + F11*F21*F22 - F21*F21*F12)
    if num * den < 0.0:
        return False

    return True


# =========================
# Residuals & Scoring
# =========================

@njit(cache=True, fastmath=True)
def sampson_residuals_sq_numba(F: np.ndarray, src: np.ndarray, dst: np.ndarray, out_sq: np.ndarray):
    N = src.shape[0]
    for i in range(N):
        x, y = src[i,0], src[i,1]
        u, v = dst[i,0], dst[i,1]
        Fx0 = F[0,0]*x + F[0,1]*y + F[0,2]
        Fx1 = F[1,0]*x + F[1,1]*y + F[1,2]
        Fx2 = F[2,0]*x + F[2,1]*y + F[2,2]
        Ft0 = F[0,0]*u + F[1,0]*v + F[2,0]
        Ft1 = F[0,1]*u + F[1,1]*v + F[2,1]
        num = u*Fx0 + v*Fx1 + Fx2
        den = Fx0*Fx0 + Fx1*Fx1 + Ft0*Ft0 + Ft1*Ft1 + 1e-15
        out_sq[i] = (num*num) / den

@njit(cache=True, fastmath=True)
def _msac_score_and_mask(F: np.ndarray, src: np.ndarray, dst: np.ndarray, threshold: float,
                         residuals_sq: np.ndarray, out_mask: np.ndarray) -> (float, int):
    """
    MSAC with squared Sampson. score = sum( min(r^2, τ^2) ), inlier if r^2 < τ^2.
    """
    sampson_residuals_sq_numba(F, src, dst, residuals_sq)
    N = src.shape[0]
    tau2 = threshold * threshold
    score = 0.0
    cnt = 0
    for i in range(N):
        r2 = residuals_sq[i]
        if r2 < tau2:
            out_mask[i] = 1
            cnt += 1
            score += r2
        else:
            out_mask[i] = 0
            score += tau2
    return score, cnt

@njit(cache=True)
def _ransac_max_trials(best_inliers: int, total_points: int, min_samples: int, p_success: float) -> int:
    """
    Canonical RANSAC iteration bound:
      max_iters = ceil( log(1 - p_success) / log(1 - w**min_samples) )
    where w = inlier_ratio. Clamped to [1, 1e9].
    """
    if p_success <= 0.0 or p_success >= 1.0:
        return 1
    if best_inliers <= min_samples or total_points <= min_samples:
        return 1
    prob_all_inlier= 1.0
    for i in range(min_samples):
        prob_all_inlier *= (best_inliers-i) / (total_points-i)
    if prob_all_inlier >= 1.0:
        return 1
    num = np.log(1.0 - p_success)
    den = np.log(1.0 - prob_all_inlier)
    if den == 0.0:
        return 1
    return int(np.ceil(num / den))


@njit(cache=True)
def _choose_best_7pt_candidate_stream(F_cands: np.ndarray,
                                      src_all: np.ndarray, dst_all: np.ndarray,
                                      rng_state: np.int64,
                                      valid_mask: np.ndarray,
                                      eval_cap: int = 64) -> (np.int64, np.int64, bool):
    """
    Choose the best 7-point candidate by stream evaluation.
    Returns: (best_k, rng_state, success)
    """
    N = src_all.shape[0]
    m = eval_cap if N > eval_cap else N
    success = False
    eval_idx = np.empty(m, dtype=np.int64)
    chosen = 0
    while chosen < m:
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        j = int(rng_state % np.int64(N))
        ok = True
        for t in range(chosen):
            if eval_idx[t] == j:
                ok = False; break
        if ok:
            eval_idx[chosen] = j
            chosen += 1

    best_k = 0
    best_score = 1e300
    tau = 1e9  # cap (not really needed)
    for k in range(3):
        if valid_mask[k] == 0:
            continue
        F = F_cands[k]
        sc = 0.0
        for t in range(m):
            i = eval_idx[t]
            x, y = src_all[i,0], src_all[i,1]
            u, v = dst_all[i,0], dst_all[i,1]
            Fx0 = F[0,0]*x + F[0,1]*y + F[0,2]
            Fx1 = F[1,0]*x + F[1,1]*y + F[1,2]
            Fx2 = F[2,0]*x + F[2,1]*y + F[2,2]
            Ft0 = F[0,0]*u + F[1,0]*v + F[2,0]
            Ft1 = F[0,1]*u + F[1,1]*v + F[2,1]
            num = u*Fx0 + v*Fx1 + Fx2
            den = Fx0*Fx0 + Fx1*Fx1 + Ft0*Ft0 + Ft1*Ft1 + 1e-15
            r2 = (num*num) / den
            sc += r2 if r2 < tau else tau
        if sc < best_score:
            best_score = sc
            best_k = k
            success = True
    return np.int64(best_k), rng_state, success


@njit(cache=True)
def _fill_A_rows(A: np.ndarray, src_n: np.ndarray, dst_n: np.ndarray, M: int):
    for i in range(M):
        x1, y1 = src_n[i,0], src_n[i,1]
        x2, y2 = dst_n[i,0], dst_n[i,1]
        A[i,0] = x2*x1; A[i,1] = x2*y1; A[i,2] = x2
        A[i,3] = y2*x1; A[i,4] = y2*y1; A[i,5] = y2
        A[i,6] = x1;    A[i,7] = y1;    A[i,8] = 1.0


@njit(cache=True)
def _gather_subset_2col(src: np.ndarray, idx: np.ndarray, k: int, out: np.ndarray):
    # src: (N,2), idx: (k,), out: (>=k,2)
    for i in range(k):
        j = idx[i]
        out[i, 0] = src[j, 0]
        out[i, 1] = src[j, 1]

@njit(cache=True)
def _gram_AtA(A: np.ndarray, M: int, G: np.ndarray):
    # Compute G = A[:M,:].T @ A[:M,:] into prealloc G (9x9)
    for i in range(9):
        for j in range(9):
            s = 0.0
            for r in range(M):
                s += A[r, i] * A[r, j]
            G[i, j] = s
            
@njit(cache=True)
def _eig_smallest_vectors_AtA(A: np.ndarray, M: int, G: np.ndarray):
    """
    Fill G with A[:M,:].T @ A[:M,:], then eigendecompose.
    Returns eigenvalues ascending and eigenvectors (columns).
    """
    # G = A[:M,:].T @ A[:M,:]
    for i in range(9):
        for j in range(9):
            s = 0.0
            for r in range(M):
                s += A[r, i] * A[r, j]
            G[i, j] = s
    w, V = np.linalg.eigh(G)  # ascending eigenvalues
    return w, V  # columns of V are eigenvectors

@njit(cache=True)
def run_8point_numba_reuse(src: np.ndarray, dst: np.ndarray,
                                A_buf: np.ndarray, G_buf: np.ndarray) -> np.ndarray:
    src_n, T1 = _normalize_points_single(src)
    dst_n, T2 = _normalize_points_single(dst)
    M = src_n.shape[0]
    _fill_A_rows(A_buf, src_n, dst_n, M)
    w, V = _eig_smallest_vectors_AtA(A_buf, M, G_buf)
    vmin = V[:, 0]                    # smallest eigenvalue -> smallest singular value
    F = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        b = 3*i
        F[i,0]=vmin[b+0]; F[i,1]=vmin[b+1]; F[i,2]=vmin[b+2]
    # Defer rank-2 until acceptance if you want; for now just denormalize:
    F = _mat33_mul(_mat33_T(T2), _mat33_mul(F, T1))
    return F


@njit(cache=True)
def _draw_unique_indices_from_mask_fast(mask: np.ndarray, k: int,
                                        rng_state: np.int64, out_idx: np.ndarray) -> (np.int64, np.int64):
    """
    Sample k unique indices uniformly from positions where mask[i]==1.
    - Returns (rng_state, got) where got==k on success, <k if not enough inliers.
    - out_idx is filled with indices into the original array (0..N-1).
    """
    N = mask.shape[0]
    # Count inliers and collect their indices
    M = 0
    for i in range(N):
        if mask[i] == 1:
            M += 1
    if M < k:
        return rng_state, np.int64(M)

    inl_idx = np.empty(M, dtype=np.int64)
    t = 0
    for i in range(N):
        if mask[i] == 1:
            inl_idx[t] = i
            t += 1

    # Sample k unique positions in [0..M-1] and map back to original indices
    tmp_k = np.empty(k, dtype=np.int64)
    rng_state = _draw_k_unique(M, k, rng_state, tmp_k)
    for i in range(k):
        out_idx[i] = inl_idx[tmp_k[i]]
    return rng_state, np.int64(k)


# ================================
# PROSAC
# ================================

@njit(cache=True)
def _prosac_next_pool_size(trial: int, N: int, k: int, max_trials: int) -> int:
    if N <= k:
        return N
    step = max(1, max_trials // (N - k + 1))
    pool = k + (trial // step)
    if pool > N: pool = N
    if pool < k: pool = k
    return pool


@njit(cache=True)
def _prosac_draw_minimal(
    N: int,
    k: int,
    trial: int,
    max_trials: int,
    rng_state: np.int64,
    out_idx: np.ndarray
) -> np.int64:
    """
    Draw a PROSAC minimal sample.
    - The input correspondences must be sorted best->worst.
    - We force-include the newest element at the current pool boundary.
    - The rest (k-1) are sampled uniquely from earlier elements.
    Returns updated rng_state.
    """
    pool = _prosac_next_pool_size(trial, N, k, max_trials)

    if pool >= N:
        # pool covers everything -> uniform minimal sample
        rng_state = _draw_k_unique(N, k, rng_state, out_idx)
        return rng_state

    # Force include the boundary element (pool-1); sample k-1 from [0, pool-1)
    boundary = pool - 1
    if k == 1:
        out_idx[0] = boundary
        return rng_state

    # Sample (k-1) unique from the first (pool-1) elements
    tmp = np.empty(k - 1, dtype=np.int64)
    rng_state = _draw_k_unique(pool - 1, k - 1, rng_state, tmp)

    # Write them out
    for i in range(k - 1):
        out_idx[i] = tmp[i]
    out_idx[k - 1] = boundary
    return rng_state



# ================================
# NAPSAC
# ================================

@njit(cache=True)
def _precompute_cell_ids(pts: np.ndarray,
                         xmin: float, xmax: float, ymin: float, ymax: float,
                         rows: int, cols: int) -> np.ndarray:
    N = pts.shape[0]
    ids = np.empty(N, dtype=np.int32)
    for i in range(N):
        r, c = _point_to_cell(pts[i,0], pts[i,1], xmin, xmax, ymin, ymax, rows, cols)
        ids[i] = r * cols + c
    return ids

@njit(cache=True)
def _bbox_from_points_xy(pts: np.ndarray):
    N = pts.shape[0]
    xmin = pts[0,0]; xmax = pts[0,0]
    ymin = pts[0,1]; ymax = pts[0,1]
    for i in range(1, N):
        x = pts[i,0]; y = pts[i,1]
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    if xmax - xmin < 1e-12: xmax = xmin + 1e-12
    if ymax - ymin < 1e-12: ymax = ymin + 1e-12
    return xmin, xmax, ymin, ymax

@njit(cache=True)
def _point_to_cell(x, y, xmin, xmax, ymin, ymax, rows, cols):
    rx = (x - xmin) / (xmax - xmin)
    ry = (y - ymin) / (ymax - ymin)
    if rx < 0.0: rx = 0.0
    if rx > 1.0: rx = 1.0
    if ry < 0.0: ry = 0.0
    if ry > 1.0: ry = 1.0
    c = int(rx * cols)
    r = int(ry * rows)
    if c >= cols: c = cols - 1
    if r >= rows: r = rows - 1
    return r, c

@njit(cache=True)
def _grid_build(points: np.ndarray, rows: int, cols: int):
    N = points.shape[0]
    xmin, xmax, ymin, ymax = _bbox_from_points_xy(points)
    C = rows * cols
    counts = np.zeros(C, dtype=np.int64)
    for i in range(N):
        r, c = _point_to_cell(points[i,0], points[i,1], xmin, xmax, ymin, ymax, rows, cols)
        counts[r*cols + c] += 1
    cell_offsets = np.empty(C + 1, dtype=np.int64)
    s = 0
    for cid in range(C):
        cell_offsets[cid] = s
        s += counts[cid]
    cell_offsets[C] = s
    for cid in range(C):
        counts[cid] = 0
    cell_indices = np.empty(N, dtype=np.int64)
    for i in range(N):
        r, c = _point_to_cell(points[i,0], points[i,1], xmin, xmax, ymin, ymax, rows, cols)
        cid = r*cols + c
        pos = cell_offsets[cid] + counts[cid]
        cell_indices[pos] = i
        counts[cid] += 1
    return xmin, xmax, ymin, ymax, cell_offsets, cell_indices


@njit(cache=True)
def _collect_cell_window(
    r: int, c: int, win: int, rows: int, cols: int,
    cell_offsets: np.ndarray, cell_indices: np.ndarray,
    out_buf: np.ndarray
) -> int:
    t = 0
    r0 = r - win; c0 = c - win
    r1 = r + win; c1 = c + win
    if r0 < 0: r0 = 0
    if c0 < 0: c0 = 0
    if r1 >= rows: r1 = rows - 1
    if c1 >= cols: c1 = cols - 1
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            cid = rr*cols + cc
            a = cell_offsets[cid]; b = cell_offsets[cid+1]
            for p in range(a, b):
                out_buf[t] = cell_indices[p]
                t += 1
    return t

@njit(cache=True)
def _is_in_first_t(arr: np.ndarray, t: int, v: int) -> bool:
    for i in range(t):
        if arr[i] == v:
            return True
    return False

@njit(cache=True)
def _distinct_cell_enforcer(
    idx: int,
    chosen: int,
    chosen_cells_src: np.ndarray,  # (k,2) of (r,c) already chosen
    min_distinct_cells: int,
    rows: int, cols: int,
    src_pts: np.ndarray,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float
) -> bool:
    """
    Before we have reached min_distinct_cells, reject idx if it falls into a cell
    already used by the chosen set (keeps early picks spatially diverse).
    """
    if chosen >= min_distinct_cells:
        return True
    r, c = _point_to_cell(src_pts[idx,0], src_pts[idx,1], xmin_s, xmax_s, ymin_s, ymax_s, rows, cols)
    for i in range(chosen):
        if chosen_cells_src[i,0] == r and chosen_cells_src[i,1] == c:
            return False
    return True


@njit(cache=True)
def _prosac_prog_napsac_draw_minimal_strict(
    # data (ranked best->worst)
    src_pts: np.ndarray, dst_pts: np.ndarray,
    N: int, k: int, trial: int, max_trials: int,
    rng_state: np.int64,
    out_idx: np.ndarray,
    # grids
    rows: int, cols: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float,
    off_s: np.ndarray, ind_s: np.ndarray,
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float,
    off_d: np.ndarray, ind_d: np.ndarray,
    # prog windows + limits
    levels_win: np.ndarray,           # e.g. [0,1,2,4]
    m_local: int,                     # take at most this many local neighbors
    min_distinct_cells: int           # keep first X picks in distinct src cells
) -> (np.int64):
    pool = _prosac_next_pool_size(trial, N, k, max_trials)

    # seed = PROSAC boundary (or uniform if pool==N)
    if pool >= N:
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        seed = int(rng_state % np.int64(N))
    else:
        seed = pool - 1

    out_idx[0] = seed
    chosen = 1

    # seed cells
    rs, cs = _point_to_cell(src_pts[seed,0], src_pts[seed,1], xmin_s, xmax_s, ymin_s, ymax_s, rows, cols)
    rd, cd = _point_to_cell(dst_pts[seed,0], dst_pts[seed,1], xmin_d, xmax_d, ymin_d, ymax_d, rows, cols)

    # tmp buffers
    cand_s = np.empty(N, dtype=np.int64)
    cand_d = np.empty(N, dtype=np.int64)
    cand_int = np.empty(N, dtype=np.int64)

    # track chosen src cells for diversity enforcement
    chosen_cells_src = np.empty((k,2), dtype=np.int64)
    chosen_cells_src[0,0] = rs; chosen_cells_src[0,1] = cs

    # progressive levels (two-view ∩, limited to pool)
    local_taken = 0
    L = levels_win.shape[0]
    for li in range(L):
        if chosen >= k or local_taken >= m_local:
            break

        win = int(levels_win[li])

        # collect src & dst neighborhoods
        cnt_s = _collect_cell_window(rs, cs, win, rows, cols, off_s, ind_s, cand_s)
        cnt_d = _collect_cell_window(rd, cd, win, rows, cols, off_d, ind_d, cand_d)

        # intersect (cand_s ∩ cand_d) keeping < pool
        # simple O(cnt_s * cnt_d) since windows are small
        t = 0
        for i in range(cnt_s):
            idx = cand_s[i]
            if idx >= pool or idx == seed:
                continue
            # linear search in cand_d (small)
            found = False
            for j in range(cnt_d):
                if cand_d[j] == idx:
                    found = True
                    break
            if found:
                cand_int[t] = idx; t += 1

        if t == 0:
            continue

        # randomize picks from cand_int with uniqueness + diversity
        tries = t * 3 + 16
        used_local = np.zeros(t, dtype=np.uint8)
        while chosen < k and local_taken < m_local and tries > 0:
            tries -= 1
            rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
            j = int(rng_state % np.int64(t))
            if used_local[j] == 1:
                continue
            used_local[j] = 1
            idx = cand_int[j]
            if _is_in_first_t(out_idx, chosen, idx):
                continue
            # enforce cell diversity for the first few picks
            if not _distinct_cell_enforcer(idx, chosen, chosen_cells_src, min_distinct_cells,
                                           rows, cols, src_pts, xmin_s, xmax_s, ymin_s, ymax_s):
                continue
            # accept
            out_idx[chosen] = idx
            chosen += 1
            local_taken += 1
            # record its src cell
            r2, c2 = _point_to_cell(src_pts[idx,0], src_pts[idx,1], xmin_s, xmax_s, ymin_s, ymax_s, rows, cols)
            chosen_cells_src[chosen-1,0] = r2; chosen_cells_src[chosen-1,1] = c2

    # Fill the rest from PROSAC pool (keeps the ranking prior)
    while chosen < k:
        pick_from = pool if pool < N else N
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        idx = int(rng_state % np.int64(pick_from))
        if _is_in_first_t(out_idx, chosen, idx):
            continue
        out_idx[chosen] = idx
        chosen += 1

    return rng_state


@njit(cache=True)
def _alpha_progressive(trial: int, prosac_only_iters: int, ramp_stop: int, alpha_max: float) -> float:
    if trial < prosac_only_iters:
        return 0.0
    if trial >= ramp_stop:
        return alpha_max
    t = trial - prosac_only_iters
    T = ramp_stop - prosac_only_iters
    a = (t / T) * alpha_max
    if a < 0.0: a = 0.0
    if a > alpha_max: a = alpha_max
    return a

@njit(cache=True)
def _draw_minimal_hybrid_v2(
    # flags
    use_prosac: bool,
    use_prog_napsac: bool,
    # data
    src_pts: np.ndarray, dst_pts: np.ndarray,
    N: int, k: int,
    trial: int, max_trials: int,
    rng_state: np.int64,
    out_idx: np.ndarray,
    # grids (src)
    rows: int, cols: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float,
    off_s: np.ndarray, ind_s: np.ndarray,
    # grids (dst)
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float,
    off_d: np.ndarray, ind_d: np.ndarray,
    # prog params
    levels_win: np.ndarray,
    m_local: int,
    min_distinct_cells: int,
    # schedule params
    prosac_only_iters: int,
    ramp_stop: int,
    alpha_max: float
) -> np.int64:
    if not use_prosac and not use_prog_napsac:
        return _draw_k_unique(N, k, rng_state, out_idx)

    if use_prog_napsac:
        a = _alpha_progressive(trial, prosac_only_iters, ramp_stop, alpha_max)
        # Bernoulli choose ProgNAPSAC vs plain PROSAC
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        coin = (rng_state % np.int64(1000000)) / 1000000.0
        if coin < a:
            # ProgNAPSAC draw
            return _prosac_prog_napsac_draw_minimal_strict(
                src_pts, dst_pts, N, k, trial, max_trials, rng_state, out_idx,
                rows, cols, xmin_s, xmax_s, ymin_s, ymax_s, off_s, ind_s,
                xmin_d, xmax_d, ymin_d, ymax_d, off_d, ind_d,
                levels_win, m_local, min_distinct_cells
            )

    # fallback: PROSAC
    pool = _prosac_next_pool_size(trial, N, k, max_trials)
    if pool >= N:
        return _draw_k_unique(N, k, rng_state, out_idx)
    # force include boundary + k-1 from [0..pool-1)
    boundary = pool - 1
    if k == 1:
        out_idx[0] = boundary
        return rng_state
    tmp = np.empty(k - 1, dtype=np.int64)
    rng_state = _draw_k_unique(pool - 1, k - 1, rng_state, tmp)
    for i in range(k - 1):
        out_idx[i] = tmp[i]
    out_idx[k - 1] = boundary
    return rng_state



@njit(cache=True)
def _choose_grid_and_levels(src_pts: np.ndarray, N: int, k: int):
    # Aim for ~8–20 pts per cell near the top pool; rough heuristic:
    # total cells ≈ N / 12, square-ish grid
    total_cells = max(4, N // 12)
    side = int(np.sqrt(total_cells))
    if side < 4: side = 4
    if side > 64: side = 64  # keep it tame

    # windows that roughly double coverage
    # 0 -> 1x1, 1 -> 3x3, 2 -> 5x5, 3 -> 9x9
    # For very sparse data k=7/8, four levels is usually enough
    levels = np.array((0, 1, 2, 4))

    return side, side, levels  # rows, cols, levels

@njit(cache=True)
def _fisher_yates_partial(buf: np.ndarray, length: int, need: int, rng_state: np.int64) -> np.int64:
    """
    In-place partial shuffle of buf[0:length], so first `need` are random unique.
    Returns updated rng_state.
    """
    if need > length:
        need = length
    for i in range(need):
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        j = i + int(rng_state % np.int64(length - i))
        tmp = buf[i]; buf[i] = buf[j]; buf[j] = tmp
    return rng_state

@njit(cache=True)
def _prosac_prog_napsac_draw_minimal_strict_fast(
    # ranked data
    src_pts: np.ndarray, dst_pts: np.ndarray,
    N: int, k: int, trial: int, max_trials: int,
    rng_state: np.int64,
    out_idx: np.ndarray,
    # grids
    rows: int, cols: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float,
    off_s: np.ndarray, ind_s: np.ndarray,
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float,
    off_d: np.ndarray, ind_d: np.ndarray,
    # precomputed per-point cell ids
    cell_id_src: np.ndarray,  # (N,) int32
    cell_id_dst: np.ndarray,  # (N,) int32
    # prog params
    levels_win: np.ndarray,   # e.g. [0,1,2,4]
    m_local: int,
    min_distinct_cells: int,
    # scratch (reused every call; all length >= N unless noted)
    cand_s: np.ndarray,       # int64 (N,)
    cand_d: np.ndarray,       # int64 (N,)
    cand_int: np.ndarray,     # int64 (N,)
    chosen_cells_src: np.ndarray, # int32 (k,2)
    mark: np.ndarray,         # uint32 (N,)
    epoch_in: np.uint32       # current epoch, will be incremented and returned
) -> (np.int64, np.uint32):
    pool = _prosac_next_pool_size(trial, N, k, max_trials)

    # seed
    if pool >= N:
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        seed = int(rng_state % np.int64(N))
    else:
        seed = pool - 1

    out_idx[0] = seed
    chosen = 1

    # seed cell ids in src/dst
    cid_s_seed = cell_id_src[seed]
    cid_d_seed = cell_id_dst[seed]
    rs = cid_s_seed // cols; cs = cid_s_seed % cols
    rd = cid_d_seed // cols; cd = cid_d_seed % cols

    # track chosen src cells
    chosen_cells_src[0,0] = rs; chosen_cells_src[0,1] = cs

    L = levels_win.shape[0]
    local_taken = 0
    epoch = epoch_in + np.uint32(1)
    if epoch == 0:
        epoch = np.uint32(1)  # wrap guard

    for li in range(L):
        if chosen >= k or local_taken >= m_local:
            break
        win = int(levels_win[li])

        # collect src window
        cnt_s = _collect_neighbors_into(rs, cs, win, rows, cols, off_s, ind_s, cand_s)
        # collect dst window and mark them for O(1) membership tests
        cnt_d = _collect_neighbors_into(rd, cd, win, rows, cols, off_d, ind_d, cand_d)

        # mark dst candidates within pool
        for j in range(cnt_d):
            idx = cand_d[j]
            if idx < pool:
                mark[idx] = epoch

        # build intersection into cand_int (filtered by pool and != seed)
        t = 0
        for i in range(cnt_s):
            idx = cand_s[i]
            if idx == seed or idx >= pool:
                continue
            if mark[idx] == epoch:
                cand_int[t] = idx
                t += 1

        if t == 0:
            continue

        # randomize first `need` = min(remaining local quota, t)
        need = m_local - local_taken
        if need > (k - chosen):
            need = (k - chosen)
        if need > t:
            need = t
        rng_state = _fisher_yates_partial(cand_int, t, need, rng_state)

        # take first `need` respecting distinct-cell constraint
        take = 0
        for q in range(need):
            idx = cand_int[q]
            # uniqueness vs already chosen minimal set
            unique = True
            for u in range(chosen):
                if out_idx[u] == idx:
                    unique = False
                    break
            if not unique:
                continue
            # distinct cell (for early picks)
            if chosen < min_distinct_cells:
                csrc = cell_id_src[idx]
                r2 = csrc // cols; c2 = csrc % cols
                ok_cell = True
                for u in range(chosen):
                    if chosen_cells_src[u,0] == r2 and chosen_cells_src[u,1] == c2:
                        ok_cell = False; break
                if not ok_cell:
                    continue
                chosen_cells_src[chosen,0] = r2; chosen_cells_src[chosen,1] = c2
            # accept
            out_idx[chosen] = idx
            chosen += 1
            take += 1
            if chosen >= k:
                break
        local_taken += take

        # clear marks we set (cheap: just bump epoch at next level)

    # fill the rest from PROSAC pool (fast)
    while chosen < k:
        pick_from = pool if pool < N else N
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        idx = int(rng_state % np.int64(pick_from))
        # uniqueness check (k<=8, linear is fine)
        ok = True
        for u in range(chosen):
            if out_idx[u] == idx:
                ok = False; break
        if not ok:
            continue
        out_idx[chosen] = idx
        chosen += 1

    return rng_state, epoch

@njit(cache=True)
def _draw_minimal_hybrid_v3_fast(
    # flags
    use_prosac: bool,
    use_prog_napsac: bool,
    # data
    src_pts: np.ndarray, dst_pts: np.ndarray,
    N: int, k: int,
    trial: int, max_trials: int,
    rng_state: np.int64,
    out_idx: np.ndarray,
    # grids
    rows: int, cols: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float, off_s: np.ndarray, ind_s: np.ndarray,
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float, off_d: np.ndarray, ind_d: np.ndarray,
    # cell ids
    cell_id_src: np.ndarray, cell_id_dst: np.ndarray,
    # prog params
    levels_win: np.ndarray, m_local: int, min_distinct_cells: int,
    # schedule
    prosac_only_iters: int, ramp_stop: int, alpha_max: float,
    # scratch
    cand_s: np.ndarray, cand_d: np.ndarray, cand_int: np.ndarray,
    chosen_cells_src: np.ndarray, mark: np.ndarray,
    epoch_in: np.uint32
) -> (np.int64, np.uint32):
    if not use_prosac and not use_prog_napsac:
        rng_state = _draw_k_unique(N, k, rng_state, out_idx)
        return rng_state, epoch_in

    # decide whether to try ProgNAPSAC this trial
    a = 0.0
    if use_prog_napsac:
        if trial < prosac_only_iters:
            a = 0.0
        elif trial >= ramp_stop:
            a = alpha_max
        else:
            a = ((trial - prosac_only_iters) / (ramp_stop - prosac_only_iters)) * alpha_max

    # coin flip
    use_local_now = use_prog_napsac and (a > 0.0)
    if use_local_now:
        # simple float coin using RNG without extra calls
        tmp = (rng_state & np.int64(0xFFFF)) / 65535.0
        # advance rng
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        if tmp < a:
            rng_state, epoch_out = _prosac_prog_napsac_draw_minimal_strict_fast(
                src_pts, dst_pts, N, k, trial, max_trials, rng_state, out_idx,
                rows, cols, xmin_s, xmax_s, ymin_s, ymax_s, off_s, ind_s,
                xmin_d, xmax_d, ymin_d, ymax_d, off_d, ind_d,
                cell_id_src, cell_id_dst,
                levels_win, m_local, min_distinct_cells,
                cand_s, cand_d, cand_int, chosen_cells_src, mark, epoch_in
            )
            return rng_state, epoch_out

    # fallback: PROSAC (force boundary + k-1 within pool)
    if use_prosac:
        pool = _prosac_next_pool_size(trial, N, k, max_trials)
        if pool >= N:
            rng_state = _draw_k_unique(N, k, rng_state, out_idx)
            return rng_state, epoch_in
        if k == 1:
            out_idx[0] = pool - 1
            return rng_state, epoch_in
        tmpk = np.empty(k - 1, dtype=np.int64)  # tiny; could be reused too if you want
        rng_state = _draw_k_unique(pool - 1, k - 1, rng_state, tmpk)
        for i in range(k - 1):
            out_idx[i] = tmpk[i]
        out_idx[k - 1] = pool - 1
        return rng_state, epoch_in

    # uniform (unlikely path)
    rng_state = _draw_k_unique(N, k, rng_state, out_idx)
    return rng_state, epoch_in

# -----------------------------
# Global minimal sampling + estimate
# -----------------------------
@njit(cache=True)
def _global_sample_and_estimate_fast(
    # data
    src_all: np.ndarray, dst_all: np.ndarray,
    min_samples: int, use_seven_point: bool,
    rng_state: np.int64,
    # tiny work buffers
    idx_buf: np.ndarray, pts1_min: np.ndarray, pts2_min: np.ndarray,
    # solvers' buffers
    A_buf: np.ndarray, G_buf: np.ndarray,
    # temporaries
    valid_mask: np.ndarray,
    # NEW: hybrid sampling controls + grids
    use_prosac: bool, use_prog_napsac: bool, trial: int, max_trials: int,
    rows: int, cols: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float, off_s: np.ndarray, ind_s: np.ndarray,
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float, off_d: np.ndarray, ind_d: np.ndarray,
    # NEW: precomputed cell ids
    cell_id_src: np.ndarray, cell_id_dst: np.ndarray,
    # Prog params + schedule
    levels_win: np.ndarray, m_local: int, min_distinct_cells: int,
    prosac_only_iters: int, ramp_stop: int, alpha_max: float,
    # NEW: scratch (preallocated once, sizes: N except chosen_cells_src (k,2))
    cand_s: np.ndarray, cand_d: np.ndarray, cand_int: np.ndarray,
    chosen_cells_src: np.ndarray, mark: np.ndarray,
    epoch_in: np.uint32
) -> (np.ndarray, np.int64, np.uint32, bool):
    N = src_all.shape[0]

    # --- FAST hybrid sampler
    rng_state, epoch_out = _draw_minimal_hybrid_v3_fast(
        use_prosac, use_prog_napsac,
        src_all, dst_all, N, min_samples,
        trial, max_trials, rng_state, idx_buf,
        rows, cols,
        xmin_s, xmax_s, ymin_s, ymax_s, off_s, ind_s,
        xmin_d, xmax_d, ymin_d, ymax_d, off_d, ind_d,
        cell_id_src, cell_id_dst,
        levels_win, m_local, min_distinct_cells,
        prosac_only_iters, ramp_stop, alpha_max,
        cand_s, cand_d, cand_int, chosen_cells_src, mark, epoch_in
    )

    # --- estimate
    if min_samples == 7 and use_seven_point:
        _gather_subset_2col(src_all, idx_buf, 7, pts1_min)
        _gather_subset_2col(dst_all, idx_buf, 7, pts2_min)
        F_cands = run_7point_numba_reuse(pts1_min[:7, :], pts2_min[:7, :], A_buf, G_buf)
        for k in range(3):
            valid_mask[k] = 1 if _rfc_check(F_cands[k]) else 0
        best_k, rng_state, success = _choose_best_7pt_candidate_stream(
            F_cands, src_all, dst_all, rng_state, valid_mask, 64
        )
        if not success:
            return np.zeros((3,3), dtype=np.float64), rng_state, epoch_out, False
        return F_cands[best_k], rng_state, epoch_out, True

    _gather_subset_2col(src_all, idx_buf, 8, pts1_min)
    _gather_subset_2col(dst_all, idx_buf, 8, pts2_min)
    F_est = run_8point_numba_reuse(pts1_min[:8, :], pts2_min[:8, :], A_buf, G_buf)
    return F_est, rng_state, epoch_out, True

@njit(cache=True)
def _global_sample_and_estimate(
    # data
    src_all: np.ndarray,
    dst_all: np.ndarray,
    min_samples: int,
    use_seven_point: bool,
    rng_state: np.int64,
    # tiny work buffers (reused by caller)
    idx_buf: np.ndarray,
    pts1_min: np.ndarray,
    pts2_min: np.ndarray,
    # reuse solvers' work buffers
    A_buf: np.ndarray,
    G_buf: np.ndarray,
    # temporaries
    valid_mask: np.ndarray,
    # --- NEW: hybrid sampling controls ---
    use_prosac: bool,
    use_prog_napsac: bool,
    trial: int,
    max_trials: int,
    # src grid
    rows_s: int, cols_s: int,
    xmin_s: float, xmax_s: float, ymin_s: float, ymax_s: float,
    off_s: np.ndarray, ind_s: np.ndarray,
    # dst grid
    xmin_d: float, xmax_d: float, ymin_d: float, ymax_d: float,
    off_d: np.ndarray, ind_d: np.ndarray,
    # Progressive NAPSAC params
    levels_win: np.ndarray,
    m_local: int,
    min_distinct_cells: int,
    # schedule params for adaptive mixing
    prosac_only_iters: int,
    ramp_stop: int,
    alpha_max: float
) -> (np.ndarray, np.int64, bool):
    """
    Draw a minimal subset via adaptive PROSAC / Progressive-NAPSAC hybrid,
    estimate F (7-pt chooses best candidate via streaming), and return:
      (F_est, rng_state, success)
    """
    N = src_all.shape[0]

    # ---- Hybrid sampler (PROSAC + Progressive NAPSAC with two-view locality) ----
    rng_state = _draw_minimal_hybrid_v2(
        use_prosac, use_prog_napsac,
        src_all, dst_all, N, min_samples,
        trial, max_trials, rng_state, idx_buf,
        # src grid
        rows_s, cols_s, xmin_s, xmax_s, ymin_s, ymax_s, off_s, ind_s,
        # dst grid
        xmin_d, xmax_d, ymin_d, ymax_d, off_d, ind_d,
        # progressive params
        levels_win, m_local, min_distinct_cells,
        # schedule
        prosac_only_iters, ramp_stop, alpha_max
    )

    # ---- Gather minimal subset ----
    if min_samples == 7 and use_seven_point:
        _gather_subset_2col(src_all, idx_buf, 7, pts1_min)
        _gather_subset_2col(dst_all, idx_buf, 7, pts2_min)

        # 7-point (multiple candidates), reuse buffers
        F_cands = run_7point_numba_reuse(pts1_min[:7, :], pts2_min[:7, :], A_buf, G_buf)

        # RFC filter for each candidate
        for k in range(3):
            valid_mask[k] = 1 if _rfc_check(F_cands[k]) else 0

        # Stream-evaluate to pick best candidate (still uses rng_state)
        best_k, rng_state, success = _choose_best_7pt_candidate_stream(
            F_cands, src_all, dst_all, rng_state, valid_mask, 64
        )
        if not success:
            return np.zeros((3,3), dtype=np.float64), rng_state, False
        return F_cands[best_k], rng_state, True

    # 8-point minimal (single estimate)
    _gather_subset_2col(src_all, idx_buf, 8, pts1_min)
    _gather_subset_2col(dst_all, idx_buf, 8, pts2_min)
    F_est = run_8point_numba_reuse(pts1_min[:8, :], pts2_min[:8, :], A_buf, G_buf)
    return F_est, rng_state, True


# -----------------------------
# Local optimization (LO-RANSAC inner loop)
# -----------------------------
@njit(cache=True)
def _local_optimize(
    F_init: np.ndarray,
    src_all: np.ndarray,
    dst_all: np.ndarray,
    threshold: float,
    # LO settings
    lo_enabled: bool,
    lo_iters: int,
    lo_thresh_mul: float,
    lo_min_inliers: int,
    lo_use_nonminimal: bool,
    lo_sample_size: int,
    use_seven_point: bool,
    # persistent scratch
    residuals: np.ndarray,
    tmp_mask: np.ndarray,
    mask_local: np.ndarray,
    # RNG + sampling scratch
    rng_state: np.int64,
    idx_buf: np.ndarray,
    pts1_lo: np.ndarray,
    pts2_lo: np.ndarray,
    # solvers' buffers
    A_buf: np.ndarray,
    G_buf: np.ndarray,
    valid_mask: np.ndarray
) -> (np.ndarray, float, int, np.int64):
    """
    Runs the LO block starting from F_init. Returns (F_local_best, score_local_best, inliers_local_best, rng_state).
    Encapsulates: tighter threshold, LO iterations with (non-)minimal subsets, and optional final refit.
    """
    N = src_all.shape[0]

    # Tighter LO threshold
    lo_thr = threshold * lo_thresh_mul
    if lo_thr <= 0.0:
        lo_thr = threshold

    # Baseline under LO threshold
    F_local = F_init
    score_local, inl_count_local = _msac_score_and_mask(
        F_local, src_all, dst_all, lo_thr, residuals, tmp_mask
    )
    # snapshot baseline mask
    for i in range(N):
        mask_local[i] = tmp_mask[i]

    if lo_enabled and inl_count_local >= lo_min_inliers:
        degen_count = 0
        for _ in range(lo_iters):
            # decide LO subset size
            k_lo = inl_count_local
            if lo_use_nonminimal:
                if k_lo > lo_sample_size:
                    k_lo = lo_sample_size
                if k_lo < 7:
                    break
            else:
                # minimal LO
                k_lo = 7 if (use_seven_point and lo_min_inliers >= 7) else 8

            rng_state, got = _draw_unique_indices_from_mask_fast(mask_local, k_lo, rng_state, idx_buf)
            if got < k_lo:
                break
            if degen_count > 8:
                break

            _gather_subset_2col(src_all, idx_buf, k_lo, pts1_lo)
            _gather_subset_2col(dst_all, idx_buf, k_lo, pts2_lo)

            # estimate on LO subset
            if k_lo >= 8:
                F_try = run_8point_numba_reuse(pts1_lo[:k_lo, :], pts2_lo[:k_lo, :], A_buf, G_buf)
            else:  # k_lo == 7
                Fc = run_7point_numba_reuse(pts1_lo[:7, :], pts2_lo[:7, :], A_buf, G_buf)
                for k in range(3):
                    valid_mask[k] = 1 if _rfc_check(Fc[k]) else 0
                best_k2, rng_state, success = _choose_best_7pt_candidate_stream(
                    Fc, src_all, dst_all, rng_state, valid_mask, 64
                )
                if not success:
                    degen_count += 1
                    continue
                F_try = Fc[best_k2]

            # evaluate under LO threshold
            score_try, inl_try = _msac_score_and_mask(F_try, src_all, dst_all, lo_thr, residuals, tmp_mask)
            if score_try < score_local:
                F_local = F_try
                score_local = score_try
                inl_count_local = inl_try
                for i in range(N):
                    mask_local[i] = tmp_mask[i]

        # Optional refit on LO inliers using global threshold (and 8-point if enough)
        score_tmp, inl_tmp = _msac_score_and_mask(F_local, src_all, dst_all, threshold, residuals, tmp_mask)
        if inl_tmp >= 8:
            M = inl_tmp
            pts1_inl = np.empty((M, 2), dtype=np.float64)
            pts2_inl = np.empty((M, 2), dtype=np.float64)
            t = 0
            for i in range(N):
                if tmp_mask[i] == 1:
                    pts1_inl[t, 0] = src_all[i, 0]; pts1_inl[t, 1] = src_all[i, 1]
                    pts2_inl[t, 0] = dst_all[i, 0]; pts2_inl[t, 1] = dst_all[i, 1]
                    t += 1
            F_local = run_8point_numba_reuse_dynamic(pts1_inl, pts2_inl, A_buf, G_buf)
            score_local, inl_count_local = _msac_score_and_mask(
                F_local, src_all, dst_all, threshold, residuals, tmp_mask
            )

    return F_local, score_local, inl_count_local, rng_state


@njit(cache=True)
def _collect_neighbors_into(
    center_r: int, center_c: int, win: int,
    rows: int, cols: int,
    cell_offsets: np.ndarray, cell_indices: np.ndarray,
    out_buf: np.ndarray
) -> int:
    """
    Collect indices from cells in the window [r-win..r+win] x [c-win..c+win] into out_buf.
    Returns number of collected indices.
    """
    t = 0
    r0 = center_r - win
    r1 = center_r + win
    c0 = center_c - win
    c1 = center_c + win
    if r0 < 0: r0 = 0
    if c0 < 0: c0 = 0
    if r1 >= rows: r1 = rows - 1
    if c1 >= cols: c1 = cols - 1

    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            cid = rr * cols + cc
            a = cell_offsets[cid]
            b = cell_offsets[cid + 1]
            # copy
            for p in range(a, b):
                out_buf[t] = cell_indices[p]
                t += 1
    return t

# -----------------------------
# Main loop
# -----------------------------
@njit(cache=True)
def ransac_fundamental_loransac_numba_refactored(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 1.0,
    min_samples: int = 7,
    max_trials: int = 2000,
    p_success: float = 0.99,
    use_seven_point: bool = True,
    seed: int = 0,
    # --- LO params ---
    lo_enabled: bool = True,
    lo_iters: int = 24,
    lo_thresh_mul: float = 0.85,
    lo_min_inliers: int = 12,
    lo_use_nonminimal: bool = True,
    lo_sample_size: int = 32,
    # NEW: samplers
    use_prosac: bool = True,
    use_prog_napsac: bool = False,
    grid_rows: int = 16,
    grid_cols: int = 16,
    # Progressive windows in cells; e.g., [0,1,2,4]
    levels_win: np.ndarray = np.array((0,1,2,4))
):
    N = src.shape[0]

    F_best = np.zeros((3,3), dtype=np.float64)
    inliers_best = np.zeros(N, dtype=np.uint8)
    best_inliers_count = -1
    best_score = 1e300

    # persistent scratch
    residuals = np.empty(N, dtype=np.float64)
    tmp_mask = np.empty(N, dtype=np.uint8)
    mask_local = np.empty(N, dtype=np.uint8)
    valid_mask = np.zeros(3, dtype=np.uint8)

    # buffers
    cap = max(lo_sample_size, min_samples)
    pts1_min = np.empty((8, 2), dtype=np.float64)
    pts2_min = np.empty((8, 2), dtype=np.float64)
    pts1_lo  = np.empty((cap, 2), dtype=np.float64)
    pts2_lo  = np.empty((cap, 2), dtype=np.float64)
    A_buf    = np.empty((cap, 9), dtype=np.float64)
    G_buf    = np.empty((9, 9), dtype=np.float64)
    idx_cap  = lo_sample_size if lo_sample_size > min_samples else min_samples
    idx_buf  = np.empty(idx_cap, dtype=np.int64)
    # Buffers done
    rng_state = np.int64(seed + 1)
    adaptive_cap = max_trials
    trials = 0

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)
    # heuristics (safe defaults)
    grid_rows = 16; grid_cols = 16
    levels_win = np.array((0,1,2,4))          # 1x1, 3x3, 5x5, 9x9
    m_local = 3                                # take at most 3 local neighbors
    min_distinct_cells = 3                     # diverse first 3 selections
    prosac_only_iters = max(64, min(512, max_trials//8))
    ramp_stop = max(prosac_only_iters + 256, prosac_only_iters + max_trials//4)
    alpha_max = 0.8                            # cap ProgNAPSAC usage

    # build grids on src & dst
    xmin_s,xmax_s,ymin_s,ymax_s, off_s, ind_s = _grid_build(src64, grid_rows, grid_cols)
    xmin_d,xmax_d,ymin_d,ymax_d, off_d, ind_d = _grid_build(dst64, grid_rows, grid_cols)

    # precompute cell ids
    cell_id_src = _precompute_cell_ids(src64, xmin_s, xmax_s, ymin_s, ymax_s, grid_rows, grid_cols)
    cell_id_dst = _precompute_cell_ids(dst64, xmin_d, xmax_d, ymin_d, ymax_d, grid_rows, grid_cols)

    # allocate scratch (reused every iteration)
    cand_s  = np.empty(N, dtype=np.int64)
    cand_d  = np.empty(N, dtype=np.int64)
    cand_int= np.empty(N, dtype=np.int64)
    chosen_cells_src = np.empty((min_samples, 2), dtype=np.int32)  # small (k,2)
    mark    = np.zeros(N, dtype=np.uint32)
    epoch   = np.uint32(1)


    while trials < max_trials and trials < adaptive_cap:
        trials += 1

        # --- (1) GLOBAL minimal sample + estimate
        # GLOBAL minimal sample + estimate (now with PROSAC)
        F_est, rng_state, epoch, ok = _global_sample_and_estimate_fast(
            src64, dst64, min_samples, use_seven_point, rng_state,
            idx_buf, pts1_min, pts2_min, A_buf, G_buf, valid_mask,
            use_prosac, use_prog_napsac, trials-1, max_trials,
            grid_rows, grid_cols,
            xmin_s, xmax_s, ymin_s, ymax_s, off_s, ind_s,
            xmin_d, xmax_d, ymin_d, ymax_d, off_d, ind_d,
            cell_id_src, cell_id_dst,
            levels_win, m_local, min_distinct_cells,
            prosac_only_iters, ramp_stop, alpha_max,
            cand_s, cand_d, cand_int, chosen_cells_src, mark, epoch
        )
        if not ok:
            continue

        # --- Global scoring
        score, inl_count = _msac_score_and_mask(F_est, src64, dst64, threshold, residuals, tmp_mask)
        if score < best_score:
            # --- (2) LOCAL OPTIMIZATION block
            F_local, score_local, inl_count_local, rng_state = _local_optimize(
                F_est, src64, dst64, threshold,
                lo_enabled, lo_iters, lo_thresh_mul, lo_min_inliers,
                lo_use_nonminimal, lo_sample_size, use_seven_point,
                residuals, tmp_mask, mask_local,
                rng_state, idx_buf, pts1_lo, pts2_lo,
                A_buf, G_buf, valid_mask
            )

            # Accept improvement (after LO / or just initial improvement)
            if score_local < best_score:
                best_score = score_local
                best_inliers_count = inl_count_local
                F_best[:, :] = F_local
                for i in range(N):
                    inliers_best[i] = tmp_mask[i]
                adaptive_cap = min(
                    max_trials,
                    _ransac_max_trials(best_inliers_count, N, min_samples, p_success)
                )

    # Final optional global refit on best inliers (8-point), if enough
    if best_inliers_count >= max(8, min_samples):
        M = best_inliers_count
        pts1_inl = np.empty((M,2), dtype=np.float64)
        pts2_inl = np.empty((M,2), dtype=np.float64)
        t = 0
        for i in range(N):
            if inliers_best[i] == 1:
                pts1_inl[t,0] = src64[i,0]; pts1_inl[t,1] = src64[i,1]
                pts2_inl[t,0] = dst64[i,0]; pts2_inl[t,1] = dst64[i,1]
                t += 1
        F_best = run_8point_numba_reuse_dynamic(pts1_inl, pts2_inl, A_buf, G_buf)
        best_score, best_inliers_count = _msac_score_and_mask(
            F_best, src64, dst64, threshold, residuals, inliers_best
        )

    return F_best, inliers_best, best_inliers_count, best_score, trials

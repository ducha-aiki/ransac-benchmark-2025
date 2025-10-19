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

    # Use provided A_buf if it fits; else allocate temp (rare)
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
    Uniformly sample k unique indices from [0, N).
    - out_idx preallocated len>=k, dtype int64
    - Returns updated rng_state
    """
    visited = np.zeros(N, dtype=np.uint8)
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
    Compute cubic coefficients for det(lambda * f1 + f2) == 0 (COLMAP explicit formula)
    with f1 already set to (v1 - v2).
    """
    a0,a1,a2,a3,a4,a5,a6,a7,a8 = f1[0],f1[1],f1[2],f1[3],f1[4],f1[5],f1[6],f1[7],f1[8]
    b0,b1,b2,b3,b4,b5,b6,b7,b8 = f2[0],f2[1],f2[2],f2[3],f2[4],f2[5],f2[6],f2[7],f2[8]

    t0 = a4 * a8 - a5 * a7
    t1 = a3 * a8 - a5 * a6
    t2 = a3 * a7 - a4 * a6
    t3 = b4 * b8 - b5 * b7
    t4 = b3 * b8 - b5 * b6
    t5 = b3 * b7 - b4 * b6

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
    7-point: (7,2),(7,2) -> (3,3,3) candidates (rank-2 projected & scaled)
    """
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

    # f1 = v1 - v2, f2 = v2
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
        for k in range(3):
            out[k, :, :] = 0.0
        return out

    roots = np.zeros(3, dtype=np.float64)
    n_roots = _solve_cubic_real(coeffs[0], coeffs[1], coeffs[2], coeffs[3], roots)

    for k in range(3):
        lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
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
    row1 = V[:, 0]; row2 = V[:, 1]

    F1 = np.empty((3,3), dtype=np.float64)
    F2 = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        b = 3*i
        F1[i,0]=row1[b+0]; F1[i,1]=row1[b+1]; F1[i,2]=row1[b+2]
        F2[i,0]=row2[b+0]; F2[i,1]=row2[b+1]; F2[i,2]=row2[b+2]

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
    out = np.zeros((3,3,3), dtype=np.float64)

    src_n, T1 = _normalize_points_single(src.astype(np.float64))
    dst_n, T2 = _normalize_points_single(dst.astype(np.float64))
    _fill_A_rows(A_buf, src_n, dst_n, 7)

    _gram_AtA(A_buf, 7, G_buf)
    w, V = np.linalg.eigh(G_buf)
    row1 = V[:, 0]; row2 = V[:, 1]

    F1 = np.empty((3,3), dtype=np.float64)
    F2 = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        base = 3*i
        F1[i,0]=row1[base+0]; F1[i,1]=row1[base+1]; F1[i,2]=row1[base+2]
        F2[i,0]=row2[base+0]; F2[i,1]=row2[base+1]; F2[i,2]=row2[base+2]

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
        return out

    roots = np.zeros(3, dtype=np.float64)
    n_roots = _solve_cubic_real(coeffs[0], coeffs[1], coeffs[2], coeffs[3], roots)

    for k in range(3):
        lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
        Fn = lam * F1 + (1.0 - lam) * F2
        F = _mat33_mul(_mat33_T(T2), _mat33_mul(Fn, T1))
        out[k] = F
    return out


### RFC check ###
@njit(cache=True)
def _rfc_check(F: np.ndarray) -> bool:
    F00, F01, F02 = F[0,0], F[0,1], F[0,2]
    F10, F11, F12 = F[1,0], F[1,1], F[1,2]
    F20, F21, F22 = F[2,0], F[2,1], F[2,2]

    den = (F00*F01*F20*F22 - F00*F02*F20*F21 +
           F01*F01*F21*F22 - F01*F02*F21*F21 +
           F10*F11*F20*F22 - F10*F12*F20*F21 +
           F11*F11*F21*F22 - F11*F12*F21*F21)

    # NOTE: corrected term F02*F02*F21 (previously F02*F2*F21)
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
      ceil( log(1 - p_success) / log(1 - w**min_samples) )
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
    tau = 1e9
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
    for i in range(k):
        j = idx[i]
        out[i, 0] = src[j, 0]
        out[i, 1] = src[j, 1]

@njit(cache=True)
def _gram_AtA(A: np.ndarray, M: int, G: np.ndarray):
    for i in range(9):
        for j in range(9):
            s = 0.0
            for r in range(M):
                s += A[r, i] * A[r, j]
            G[i, j] = s

@njit(cache=True)
def _eig_smallest_vectors_AtA(A: np.ndarray, M: int, G: np.ndarray):
    for i in range(9):
        for j in range(9):
            s = 0.0
            for r in range(M):
                s += A[r, i] * A[r, j]
            G[i, j] = s
    w, V = np.linalg.eigh(G)
    return w, V  # columns = eigenvectors

@njit(cache=True)
def run_8point_numba_reuse(src: np.ndarray, dst: np.ndarray,
                           A_buf: np.ndarray, G_buf: np.ndarray) -> np.ndarray:
    src_n, T1 = _normalize_points_single(src)
    dst_n, T2 = _normalize_points_single(dst)
    M = src_n.shape[0]
    _fill_A_rows(A_buf, src_n, dst_n, M)
    w, V = _eig_smallest_vectors_AtA(A_buf, M, G_buf)
    vmin = V[:, 0]
    F = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        b = 3*i
        F[i,0]=vmin[b+0]; F[i,1]=vmin[b+1]; F[i,2]=vmin[b+2]
    F = _mat33_mul(_mat33_T(T2), _mat33_mul(F, T1))
    return F


@njit(cache=True)
def _draw_unique_indices_from_mask_fast(mask: np.ndarray, k: int,
                                        rng_state: np.int64, out_idx: np.ndarray) -> (np.int64, np.int64):
    """
    Sample k unique indices uniformly where mask[i]==1.
    Returns (rng_state, got) with got==k on success.
    """
    N = mask.shape[0]
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

    tmp_k = np.empty(k, dtype=np.int64)
    rng_state = _draw_k_unique(M, k, rng_state, tmp_k)
    for i in range(k):
        out_idx[i] = inl_idx[tmp_k[i]]
    return rng_state, np.int64(k)


# ================================
# PROSAC (only)
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
    N: int, k: int, trial: int, max_trials: int,
    rng_state: np.int64, out_idx: np.ndarray
) -> np.int64:
    """
    Draw a PROSAC minimal sample (data must be ranked best->worst).
    Force-include the current pool boundary; draw the rest from earlier items.
    """
    pool = _prosac_next_pool_size(trial, N, k, max_trials)

    if pool >= N:
        rng_state = _draw_k_unique(N, k, rng_state, out_idx)
        return rng_state

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


# -----------------------------
# Global minimal sampling + estimate (PROSAC-only)
# -----------------------------
@njit(cache=True)
def _global_sample_and_estimate(
    src_all: np.ndarray,
    dst_all: np.ndarray,
    min_samples: int,
    use_seven_point: bool,
    rng_state: np.int64,
    # tiny work buffers
    idx_buf: np.ndarray,
    pts1_min: np.ndarray,
    pts2_min: np.ndarray,
    # solver work buffers
    A_buf: np.ndarray,
    G_buf: np.ndarray,
    # temporaries
    valid_mask: np.ndarray,
    # PROSAC controls
    use_prosac: bool,
    trial: int,
    max_trials: int
) -> (np.ndarray, np.int64, bool):
    N = src_all.shape[0]

    # Draw minimal subset
    if use_prosac:
        rng_state = _prosac_draw_minimal(N, min_samples, trial, max_trials, rng_state, idx_buf)
    else:
        rng_state = _draw_k_unique(N, min_samples, rng_state, idx_buf)

    # Estimate
    if min_samples == 7 and use_seven_point:
        _gather_subset_2col(src_all, idx_buf, 7, pts1_min)
        _gather_subset_2col(dst_all, idx_buf, 7, pts2_min)
        F_cands = run_7point_numba_reuse(pts1_min[:7, :], pts2_min[:7, :], A_buf, G_buf)
        for k in range(3):
            if _rfc_check(F_cands[k]):
                valid_mask[k] = np.uint8(1)
            else:
                valid_mask[k] = np.uint8(0)
        best_k, rng_state, success = _choose_best_7pt_candidate_stream(
            F_cands, src_all, dst_all, rng_state, valid_mask, 64
        )
        if not success:
            return np.zeros((3,3), dtype=np.float64), rng_state, False
        return F_cands[best_k], rng_state, True

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
                k_lo = 7 if (use_seven_point and lo_min_inliers >= 7) else 8

            rng_state, got = _draw_unique_indices_from_mask_fast(mask_local, k_lo, rng_state, idx_buf)
            if got < k_lo:
                break
            if degen_count > 8:
                break
            _gather_subset_2col(src_all, idx_buf, k_lo, pts1_lo)
            _gather_subset_2col(dst_all, idx_buf, k_lo, pts2_lo)

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

            score_try, inl_try = _msac_score_and_mask(F_try, src_all, dst_all, lo_thr, residuals, tmp_mask)
            if score_try < score_local:
                F_local = F_try
                score_local = score_try
                inl_count_local = inl_try
                for i in range(N):
                    mask_local[i] = tmp_mask[i]

        # Optional refit on LO inliers using global threshold (8-point if enough)
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


### Refinement
@njit(cache=True)
def _quat_normalize(q):
    n = np.sqrt(q[0]*q[0] + q[1]*q[1] + q[2]*q[2] + q[3]*q[3]) + 1e-18
    return np.array((q[0]/n, q[1]/n, q[2]/n, q[3]/n), dtype=np.float64)

@njit(cache=True)
def _quat_mul(q1, q2):
    # (w,x,y,z)
    w1,x1,y1,z1 = q1[0],q1[1],q1[2],q1[3]
    w2,x2,y2,z2 = q2[0],q2[1],q2[2],q2[3]
    return np.array((
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ), dtype=np.float64)

@njit(cache=True)
def _exp_so3_to_quat(w):
    # w: (3,) rotation vector; q = [cos(|w|/2), sin(|w|/2) * (w/|w|)]
    th = np.sqrt(w[0]*w[0] + w[1]*w[1] + w[2]*w[2])
    half = 0.5 * th
    if th < 1e-12:
        s = 0.5 - (th*th)/48.0  # sin(half)/th ~ 1/2 - th^2/48
    else:
        s = np.sin(half) / th
    return _quat_normalize(np.array((np.cos(half), s*w[0], s*w[1], s*w[2]), dtype=np.float64))

@njit(cache=True)
def _quat_to_rotmat(q):
    w,x,y,z = q[0],q[1],q[2],q[3]
    xx,yy,zz = x*x, y*y, z*z
    wx,wy,wz = w*x, w*y, w*z
    xy,xz,yz = x*y, x*z, y*z
    R = np.empty((3,3), dtype=np.float64)
    R[0,0] = 1.0 - 2.0*(yy + zz)
    R[0,1] = 2.0*(xy - wz)
    R[0,2] = 2.0*(xz + wy)
    R[1,0] = 2.0*(xy + wz)
    R[1,1] = 1.0 - 2.0*(xx + zz)
    R[1,2] = 2.0*(yz - wx)
    R[2,0] = 2.0*(xz - wy)
    R[2,1] = 2.0*(yz + wx)
    R[2,2] = 1.0 - 2.0*(xx + yy)
    return R

@njit(cache=True)
def _rotmat_to_quat(R):
    # Robust-from-trace branch (w,x,y,z)
    t = R[0,0] + R[1,1] + R[2,2]
    if t > 0.0:
        s = np.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2,1] - R[1,2]) / s
        y = (R[0,2] - R[2,0]) / s
        z = (R[1,0] - R[0,1]) / s
    else:
        if R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2.0
            w = (R[2,1] - R[1,2]) / s
            x = 0.25 * s
            y = (R[0,1] + R[1,0]) / s
            z = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2.0
            w = (R[0,2] - R[2,0]) / s
            x = (R[0,1] + R[1,0]) / s
            y = 0.25 * s
            z = (R[1,2] + R[2,1]) / s
        else:
            s = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2.0
            w = (R[1,0] - R[0,1]) / s
            x = (R[0,2] + R[2,0]) / s
            y = (R[1,2] + R[2,1]) / s
            z = 0.25 * s
    return _quat_normalize(np.array((w,x,y,z), dtype=np.float64))

@njit(cache=True)
def _ffm_from_F(F):
    # SVD (rank-2 enforced outside); take U,V with det>0; sigma = s2/s1
    U, S, Vt = np.linalg.svd(F)
    U = U.copy(); V = _mat33_T(Vt)
    #if np.linalg.det(U) < 0: U[:,0] = -U[:,0]; U[:,1] = -U[:,1]; U[:,2] = -U[:,2]
    #if np.linalg.det(V) < 0: V[:,0] = -V[:,0]; V[:,1] = -V[:,1]; V[:,2] = -V[:,2]
    if np.linalg.det(U) < 0: U[:,2] = -U[:,2]
    if np.linalg.det(V) < 0: V[:,2] = -V[:,2]

    qU = _rotmat_to_quat(U)
    qV = _rotmat_to_quat(V)
    s1 = S[0] if S[0] != 0.0 else 1.0
    sigma = S[1] / s1
    return qU, qV, sigma

@njit(cache=True)
def _ffm_to_F(qU, qV, sigma):
    U = _quat_to_rotmat(qU)
    V = _quat_to_rotmat(qV)
    # F = u0 v0^T + sigma * u1 v1^T (Bartoli–Sturm)
    F = np.empty((3,3), dtype=np.float64)
    u0 = U[:,0]; u1 = U[:,1]
    v0 = V[:,0]; v1 = V[:,1]
    for i in range(3):
        for j in range(3):
            F[i,j] = u0[i]*v0[j] + sigma * u1[i]*v1[j]
    return F

@njit(cache=True)
def _ffm_step(qU, qV, sigma, dp):
    # dp: [wU(3), wV(3), d_sigma]
    dqu = _exp_so3_to_quat(dp[0:3])
    dqv = _exp_so3_to_quat(dp[3:6])
    qU_new = _quat_mul(dqu, qU)   # left-multiply
    qV_new = _quat_mul(dqv, qV)
    qU_new = _quat_normalize(qU_new)
    qV_new = _quat_normalize(qV_new)
    return qU_new, qV_new, sigma + dp[6]

@njit(cache=True)
def _sampson_errors(F, src, dst, out_r):
    # out_r: **signed** Sampson residual
    N = src.shape[0]
    for i in range(N):
        x, y = src[i,0], src[i,1]
        u, v = dst[i,0], dst[i,1]
        Fx0 = F[0,0]*x + F[0,1]*y + F[0,2]
        Fx1 = F[1,0]*x + F[1,1]*y + F[1,2]
        Fx2 = F[2,0]*x + F[2,1]*y + F[2,2]
        Ft0 = F[0,0]*u + F[1,0]*v + F[2,0]
        Ft1 = F[0,1]*u + F[1,1]*v + F[2,1]
        num = u*Fx0 + v*Fx1 + Fx2             # u^T F x (signed)
        den = Fx0*Fx0 + Fx1*Fx1 + Ft0*Ft0 + Ft1*Ft1 + 1e-15
        out_r[i] = num / np.sqrt(den)         # **keep the sign**

@njit(cache=True)
def _huber_weights(r, delta, out_w):
    N = r.shape[0]
    for i in range(N):
        a = abs(r[i])
        if a <= delta:
            out_w[i] = 1.0
        else:
            out_w[i] = delta / (a + 1e-18)

@njit(cache=True)
def _cauchy_weights(r, c, out_w):
    # ρ(r) = c^2 * log(1 + (r/c)^2); w = ψ/|r| = 1 / (1 + (r/c)^2)
    c2 = c*c
    for i in range(r.shape[0]):
        a = r[i]
        out_w[i] = 1.0 / (1.0 + (a*a)/(c2 + 1e-18))


@njit(cache=True)
def _chol7(Ain):
    # Cholesky factor L for symmetric positive definite A (7x7)
    # Returns lower-triangular L such that A = L L^T
    L = np.zeros((7,7), dtype=np.float64)
    for i in range(7):
        for j in range(i+1):
            s = Ain[i,j]
            for k in range(j):
                s -= L[i,k]*L[j,k]
            if i == j:
                # jitter for numerical safety
                if s <= 1e-18:
                    s = s + 1e-12
                L[i,j] = np.sqrt(s)
            else:
                L[i,j] = s / (L[j,j] + 1e-18)
    return L

@njit(cache=True)
def _solve7_spd(A, b):
    # Solve A x = b where A is SPD via Cholesky (A = L L^T)
    L = _chol7(A)
    # forward solve L y = b
    y = np.zeros(7, dtype=np.float64)
    for i in range(7):
        s = b[i]
        for k in range(i):
            s -= L[i,k]*y[k]
        y[i] = s / (L[i,i] + 1e-18)
    # back solve L^T x = y
    x = np.zeros(7, dtype=np.float64)
    for i in range(6, -1, -1):
        s = y[i]
        for k in range(i+1, 7):
            s -= L[k,i]*x[k]
        x[i] = s / (L[i,i] + 1e-18)
    return x

@njit(cache=True)
def _norm2D(pts):
    cx = 0.0; cy = 0.0
    N = pts.shape[0]
    for i in range(N):
        cx += pts[i,0]; cy += pts[i,1]
    invN = 1.0 / float(N)
    cx *= invN; cy *= invN
    md = 0.0
    for i in range(N):
        dx = pts[i,0] - cx; dy = pts[i,1] - cy
        md += np.sqrt(dx*dx + dy*dy)
    md *= invN
    s = np.sqrt(2.0) / (md + 1e-18)
    T = np.empty((3,3), dtype=np.float64)
    T[0,0]=s; T[0,1]=0.0; T[0,2]=-s*cx
    T[1,0]=0.0; T[1,1]=s; T[1,2]=-s*cy
    T[2,0]=0.0; T[2,1]=0.0; T[2,2]=1.0
    q = np.empty_like(pts)
    for i in range(N):
        q[i,0] = s*pts[i,0] - s*cx
        q[i,1] = s*pts[i,1] - s*cy
    return T, q

@njit(cache=True)
def _inv33(A):
    det = (A[0,0]*(A[1,1]*A[2,2]-A[1,2]*A[2,1])
            -A[0,1]*(A[1,0]*A[2,2]-A[1,2]*A[2,0])
            +A[0,2]*(A[1,0]*A[2,1]-A[1,1]*A[2,0]))
    if abs(det) < 1e-18:
        I = np.zeros((3,3), dtype=np.float64)
        I[0,0]=1.0; I[1,1]=1.0; I[2,2]=1.0
        return I
    inv = np.empty((3,3), dtype=np.float64)
    inv[0,0] =  (A[1,1]*A[2,2]-A[1,2]*A[2,1])/det
    inv[0,1] = -(A[0,1]*A[2,2]-A[0,2]*A[2,1])/det
    inv[0,2] =  (A[0,1]*A[1,2]-A[0,2]*A[1,1])/det
    inv[1,0] = -(A[1,0]*A[2,2]-A[1,2]*A[2,0])/det
    inv[1,1] =  (A[0,0]*A[2,2]-A[0,2]*A[2,0])/det
    inv[1,2] = -(A[0,0]*A[1,2]-A[0,2]*A[1,0])/det
    inv[2,0] =  (A[1,0]*A[2,1]-A[1,1]*A[2,0])/det
    inv[2,1] = -(A[0,0]*A[2,1]-A[0,1]*A[2,0])/det
    inv[2,2] =  (A[0,0]*A[1,1]-A[0,1]*A[1,0])/det
    return inv

@njit(cache=True)
def _sigmoid(x):
    # stable-ish
    if x >= 0:
        ex = np.exp(-x)
        return 1.0 / (1.0 + ex)
    else:
        ex = np.exp(x)
        return ex / (1.0 + ex)

@njit(cache=True)
def _clamp01(a):
    if a < 1e-6: return 1e-6
    if a > 1.0 - 1e-6: return 1.0 - 1e-6
    return a

@njit(cache=True)
def _logit(p):
    p = _clamp01(p)
    return np.log(p / (1.0 - p))

@njit(cache=True)
def _apply_step(qU, qV, s, dp):
    # rotations via SO(3) exp (left-multiply), s additively
    dqu = _exp_so3_to_quat(dp[0:3])
    dqv = _exp_so3_to_quat(dp[3:6])
    qU_new = _quat_mul(dqu, qU)
    qV_new = _quat_mul(dqv, qV)
    qU_new = _quat_normalize(qU_new)
    qV_new = _quat_normalize(qV_new)
    s_new = s + dp[6]
    return qU_new, qV_new, s_new

@njit(cache=True)
def _cauchy_weights(r, c, out_w):
    c2 = c*c
    N = r.shape[0]
    for i in range(N):
        a = r[i]
        out_w[i] = 1.0 / (1.0 + (a*a)/(c2 + 1e-18))

@njit(cache=True)
def _cost(rr, ww):
    s = 0.0
    for i in range(rr.shape[0]):
        s += ww[i] * rr[i] * rr[i]
    return 0.5 * s


@njit(cache=True)
def _eps_sig(sig):
    a = sig if sig >= 0.0 else -sig
    v = 1e-3 * (a if a >= 1.0 else 1.0)
    return v

@njit(cache=True)
# Cost helper
def _cost(rr, ww):
    s = 0.0
    for i in range(rr.shape[0]):
        s += ww[i] * rr[i] * rr[i]
    return 0.5 * s

# Symmetric epipolar residuals (two signed residuals per pair)
@njit(cache=True)
def _sym_epi_res(F, src, dst, out_r):
    for i in range(src.shape[0]):
        x, y = src[i,0], src[i,1]
        u, v = dst[i,0], dst[i,1]
        Fx0 = F[0,0]*x + F[0,1]*y + F[0,2]
        Fx1 = F[1,0]*x + F[1,1]*y + F[1,2]
        Fx2 = F[2,0]*x + F[2,1]*y + F[2,2]
        Ft0 = F[0,0]*u + F[1,0]*v + F[2,0]
        Ft1 = F[0,1]*u + F[1,1]*v + F[2,1]
        num = u*Fx0 + v*Fx1 + Fx2  # signed
        d2 = np.sqrt(Fx0*Fx0 + Fx1*Fx1) + 1e-18  # distance in image 2
        d1 = np.sqrt(Ft0*Ft0 + Ft1*Ft1) + 1e-18  # distance in image 1
        out_r[2*i + 0] = num / d2
        out_r[2*i + 1] = num / d1
            
@njit(cache=True)
def _refine_fundamental_bs(
    F_init: np.ndarray,
    src_inl: np.ndarray,   # (M,2)
    dst_inl: np.ndarray,   # (M,2)
    max_iters: int = 20,
    huber_delta_unused: float = 1.0  # API compat; unused
) -> np.ndarray:
    """
    LM refinement of F (Bartoli-Sturm) with:
      - Hartley normalization (internal)
      - Symmetric epipolar distance (two signed residuals per match)
      - Cauchy IRLS (MAD-scaled, fixed per outer iteration)
      - Central-diff Jacobian of those residuals
      - Diagonal (Marquardt) damping + SPD Cholesky
      - Two LM inner steps per weight update
      - Additive sigma
    Returns denormalized F.
    """
    M = src_inl.shape[0]
    if M < 8:
        return F_init

    # ----- Normalize points and F -----
    T1, srcN = _norm2D(src_inl)
    T2, dstN = _norm2D(dst_inl)
    T1_inv = _inv33(T1)
    T2_inv = _inv33(T2)
    Fhat0 = T2_inv.T @ F_init @ T1_inv  # work in normalized coords

    # ----- Init factorization -----
    qU, qV, sigma = _ffm_from_F(Fhat0)

    # Buffers for 2M residuals
    R  = np.empty(2*M, dtype=np.float64)
    RP = np.empty(2*M, dtype=np.float64)
    RM = np.empty(2*M, dtype=np.float64)
    W  = np.empty(2*M, dtype=np.float64)
    J  = np.empty((7, 2*M), dtype=np.float64)

    # FD steps
    eps_rot = 1e-3


    # Initial residuals
    Fhat = _ffm_to_F(qU, qV, sigma)
    _sym_epi_res(Fhat, srcN, dstN, R)

    lamb = -1.0  # lazy init

    # ----- IRLS outer loop -----
    for it in range(max_iters):
        # Cauchy weights from MAD(|R|) over 2M residuals
        absr = np.empty(2*M, dtype=np.float64)
        for i in range(2*M):
            a = R[i]; absr[i] = a if a >= 0.0 else -a
        # median by insertion sort (tiny arrays are OK under numba)
        for a in range(1, 2*M):
            v = absr[a]; b = a - 1
            while b >= 0 and absr[b] > v:
                absr[b+1] = absr[b]; b -= 1
            absr[b+1] = v
        med = absr[ (2*M)//2 ]
        sigma_r = 1.4826 * med
        c = max(1.2 * sigma_r, 5e-4)   # slightly tighter than before
        _cauchy_weights(R, c, W)       # fixed weights for inner steps

        # ----- two inner LM steps with fixed W -----
        for inner in range(2):
            # Central-diff Jacobian columns (wrt 7 params) for 2M residuals
            eps_sig = _eps_sig(sigma)
            for j in range(7):
                step = eps_rot if j < 6 else eps_sig
                dp = np.zeros(7, dtype=np.float64)

                # +step
                dp[j] = step
                qU_p, qV_p, sig_p = _ffm_step(qU, qV, sigma, dp)
                Fp = _ffm_to_F(qU_p, qV_p, sig_p)
                _sym_epi_res(Fp, srcN, dstN, RP)

                # -step
                dp[j] = -step
                qU_m, qV_m, sig_m = _ffm_step(qU, qV, sigma, dp)
                Fm = _ffm_to_F(qU_m, qV_m, sig_m)
                _sym_epi_res(Fm, srcN, dstN, RM)

                inv2h = 0.5 / step
                for i in range(2*M):
                    J[j, i] = (RP[i] - RM[i]) * inv2h

            # Build normal equations
            JtWr = np.zeros(7, dtype=np.float64)
            JtWJ = np.zeros((7, 7), dtype=np.float64)
            for j in range(7):
                s1 = 0.0
                colj = J[j]
                for i in range(2*M):
                    s1 += W[i] * colj[i] * R[i]
                JtWr[j] = s1
                for k in range(j + 1):
                    s2 = 0.0
                    colk = J[k]
                    for i in range(2*M):
                        s2 += W[i] * colj[i] * colk[i]
                    JtWJ[j, k] = s2
                    JtWJ[k, j] = s2

            # Diagonal (Marquardt) damping with simple outlier cap
            D = np.zeros(7, dtype=np.float64)
            maxD = 0.0; idxMax = 0
            for d in range(7):
                D[d] = JtWJ[d,d] + 1e-18
                if D[d] > maxD:
                    maxD = D[d]; idxMax = d
            secondMax = 0.0
            for d in range(7):
                if d == idxMax: continue
                if D[d] > secondMax: secondMax = D[d]
            cap = secondMax if secondMax > 0.0 else maxD
            for d in range(7):
                if D[d] > cap: D[d] = cap

            if lamb < 0.0:
                Ds = np.empty(7, dtype=np.float64)
                for d in range(7): Ds[d] = D[d]
                for a in range(1,7):
                    v = Ds[a]; b = a-1
                    while b >= 0 and Ds[b] > v:
                        Ds[b+1] = Ds[b]; b -= 1
                    Ds[b+1] = v
                medD = Ds[3]
                lamb = 1e-8 * (medD + 1e-18)
                if lamb < 1e-12: lamb = 1e-12

            cost_old = _cost(R, W)

            # LM step solve
            A = JtWJ.copy()
            for d in range(7):
                A[d,d] += lamb * D[d]
            g = JtWr.copy()
            dp = _solve7_spd(A, -g)

            # Trial
            qU_new, qV_new, sigma_new = _ffm_step(qU, qV, sigma, dp)
            Fhat_new = _ffm_to_F(qU_new, qV_new, sigma_new)
            _sym_epi_res(Fhat_new, srcN, dstN, RP)
            cost_new = _cost(RP, W)

            # Predicted reduction (diag damping): 0.5 * dp^T (λ D dp - g)
            tmp = 0.0
            for j in range(7):
                tmp += dp[j] * (lamb * D[j] * dp[j] - g[j])
            pred_red = 0.5 * (tmp if tmp > 1e-18 else 1e-18)
            rho = (cost_old - cost_new) / pred_red

            if (cost_new < cost_old) and (rho > 0.0):
                # accept
                qU, qV, sigma = qU_new, qV_new, sigma_new
                Fhat = Fhat_new
                R[:] = RP[:]

                # lambda update
                two_rho_minus1 = 2.0 * rho - 1.0
                tmpf = 1.0 - two_rho_minus1 * two_rho_minus1 * two_rho_minus1
                if tmpf < (1.0/3.0): tmpf = (1.0/3.0)
                lamb = lamb * tmpf
                if lamb < 1e-12: lamb = 1e-12

                # quick exit if tiny step
                max_dp = 0.0
                for j in range(7):
                    a = dp[j] if dp[j] >= 0.0 else -dp[j]
                    if a > max_dp: max_dp = a
                if max_dp < 1e-6:
                    break
            else:
                lamb *= 2.5
                if lamb > 1e12:
                    break
                break  # reweight next outer

        # Outer early stop on tiny relative improvement
        curr_cost = _cost(R, W)
        diff = cost_old - curr_cost
        if (diff if diff >= 0.0 else -diff) <= 1e-9 * (1.0 + cost_old):
            break

    # ----- Denormalize -----
    F_out = T2.T @ Fhat @ T1
    return F_out


###

# -----------------------------
# Main loop (PROSAC-only)
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
    # Sampling strategy
    use_prosac: bool = True,
    refine_nonlinear: bool = False,
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

    rng_state = np.int64(seed + 1)
    adaptive_cap = max_trials
    trials = 0

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)

    while trials < max_trials and trials < adaptive_cap:
        trials += 1

        # (1) global minimal sample + estimate (PROSAC or uniform)
        F_est, rng_state, ok = _global_sample_and_estimate(
            src64, dst64, min_samples, use_seven_point, rng_state,
            idx_buf, pts1_min, pts2_min, A_buf, G_buf, valid_mask,
            use_prosac, trials - 1, max_trials
        )
        if not ok:
            continue

        # (2) global scoring
        score, inl_count = _msac_score_and_mask(F_est, src64, dst64, threshold, residuals, tmp_mask)
        if score < best_score:
            # (3) LO block
            F_local, score_local, inl_count_local, rng_state = _local_optimize(
                F_est, src64, dst64, threshold,
                lo_enabled, lo_iters, lo_thresh_mul, lo_min_inliers,
                lo_use_nonminimal, lo_sample_size, use_seven_point,
                residuals, tmp_mask, mask_local,
                rng_state, idx_buf, pts1_lo, pts2_lo,
                A_buf, G_buf, valid_mask
            )
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
    # Final refit on best inliers (8-point), if enough
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
        best_score, best_inliers_count = _msac_score_and_mask(F_best, src64, dst64, threshold, residuals, inliers_best)
    # --- Non-linear refinement on best inliers (Bartoli–Sturm, IRLS Huber) ---

    refine_iters: int = 12
    refine_huber: float = threshold
    if refine_nonlinear and best_inliers_count >= max(8, min_samples):
        M = best_inliers_count
        pts1_inl = np.empty((M,2), dtype=np.float64)
        pts2_inl = np.empty((M,2), dtype=np.float64)
        t = 0
        for i in range(N):
            if inliers_best[i] == 1:
                pts1_inl[t,0] = src64[i,0]; pts1_inl[t,1] = src64[i,1]
                pts2_inl[t,0] = dst64[i,0]; pts2_inl[t,1] = dst64[i,1]
                t += 1
        F_ref = _refine_fundamental_bs(F_best, pts1_inl, pts2_inl, max_iters=refine_iters)
        # Re-evaluate with global threshold and keep if better
        score_ref, inl_ref = _msac_score_and_mask(F_ref, src64, dst64, threshold, residuals, tmp_mask)
        if score_ref <= best_score:
            F_best = F_ref
            best_score = score_ref
            best_inliers_count = inl_ref
            for i in range(N):
                inliers_best[i] = tmp_mask[i]

    return F_best, inliers_best, best_inliers_count, best_score, trials

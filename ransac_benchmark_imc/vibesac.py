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
    # use your preferred solver (robust or original); example:
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
def _build_A_N(src_n: np.ndarray, dst_n: np.ndarray) -> np.ndarray:
    N = src_n.shape[0]
    A = np.ones((N, 9), dtype=src_n.dtype)
    for i in range(N):
        x1, y1 = src_n[i, 0], src_n[i, 1]
        x2, y2 = dst_n[i, 0], dst_n[i, 1]
        A[i, 0] = x2 * x1; A[i, 1] = x2 * y1; A[i, 2] = x2
        A[i, 3] = y2 * x1; A[i, 4] = y2 * y1; A[i, 5] = y2
        A[i, 6] = x1;      A[i, 7] = y1
    return A

@njit(cache=True)
def run_8point_numba(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    src_n, T1 = _normalize_points_single(src.astype(np.float64))
    dst_n, T2 = _normalize_points_single(dst.astype(np.float64))
    A = _build_A_N(src_n, dst_n)
    U, S, Vt = np.linalg.svd(A)
    row = Vt[-1]
    F = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        base = 3 * i
        F[i, 0] = row[base + 0]; F[i, 1] = row[base + 1]; F[i, 2] = row[base + 2]
    F = _rank2_project(F)
    F = _mat33_mul(_mat33_T(T2), _mat33_mul(F, T1))
    d = F[2, 2]
    if abs(d) > 1e-8:
        F = F / (d + 1e-8)
    return F

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


# =========================
# Proper RANSAC iteration cap
# =========================


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
    # w**min_samples (stable in Numba using loop)
    prob_all_inlier= 1.0
    for i in range(min_samples):
        prob_all_inlier *= (best_inliers-i) / (total_points-i)
    if prob_all_inlier >= 1.0:
        return 1
    num = np.log(1.0 - p_success)
    den = np.log(1.0 - prob_all_inlier)
    if den == 0.0:  # extremely small (wm≈1)
        return 1
    return int(np.ceil(num / den))


@njit(cache=True)
def _draw_unique_indices_from_mask(mask: np.ndarray, k: int, rng_state: np.int64, out_idx: np.ndarray) -> (np.int64, np.int64):
    """
    Sample k unique indices uniformly from set where mask[i]==1.
    Returns (rng_state, M) where M==k if success, else M<k if not enough inliers.
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

    chosen = 0
    while chosen < k:
        rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
        j = int(rng_state % np.int64(M))
        cand = inl_idx[j]
        ok = True
        for t2 in range(chosen):
            if out_idx[t2] == cand:
                ok = False; break
        if ok:
            out_idx[chosen] = cand
            chosen += 1
    return rng_state, np.int64(k)

@njit(cache=True)
def _choose_best_7pt_candidate_stream(F_cands: np.ndarray,
                                      src_all: np.ndarray, dst_all: np.ndarray,
                                      rng_state: np.int64,
                                      eval_cap: int = 64) -> (np.int64, np.int64):
    N = src_all.shape[0]
    m = eval_cap if N > eval_cap else N

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
    return np.int64(best_k), rng_state


@njit(cache=True)
def _eigvals_2x2_sym(a11, a12, a22):
    tr = a11 + a22
    det = a11 * a22 - a12 * a12
    disc = tr*tr - 4.0*det
    if disc < 0.0: disc = 0.0
    s = np.sqrt(disc)
    l1 = 0.5*(tr + s); l2 = 0.5*(tr - s)
    if l2 > l1: l1, l2 = l2, l1
    return l1, l2

@njit(cache=True)
def _is_degenerate_minimal(pts: np.ndarray, ratio_thresh: float = 1e-3, extent_thresh: float = 1e-6) -> bool:
    k = pts.shape[0]
    cx0 = 0.0; cx1 = 0.0
    for i in range(k):
        cx0 += pts[i,0]; cx1 += pts[i,1]
    cx0 /= k; cx1 /= k
    s11 = 0.0; s22 = 0.0; s12 = 0.0
    for i in range(k):
        dx = pts[i,0]-cx0; dy = pts[i,1]-cx1
        s11 += dx*dx; s22 += dy*dy; s12 += dx*dy
    l1, l2 = _eigvals_2x2_sym(s11, s12, s22)
    tot = l1 + l2
    if tot < extent_thresh: return True
    if l1 <= 0.0: return True
    return (l2 / l1) < ratio_thresh

@njit(cache=True)
def _fill_A_rows(A: np.ndarray, src_n: np.ndarray, dst_n: np.ndarray, M: int):
    for i in range(M):
        x1, y1 = src_n[i,0], src_n[i,1]
        x2, y2 = dst_n[i,0], dst_n[i,1]
        A[i,0] = x2*x1; A[i,1] = x2*y1; A[i,2] = x2
        A[i,3] = y2*x1; A[i,4] = y2*y1; A[i,5] = y2
        A[i,6] = x1;    A[i,7] = y1;    A[i,8] = 1.0

@njit(cache=True)
def run_8point_numba_reuse(src: np.ndarray, dst: np.ndarray, A_buf: np.ndarray) -> np.ndarray:
    src_n, T1 = _normalize_points_single(src)
    dst_n, T2 = _normalize_points_single(dst)
    M = src_n.shape[0]
    _fill_A_rows(A_buf, src_n, dst_n, M)          # fill first M rows
    A = A_buf[:M, :]                               # view, no new alloc
    U, S, Vt = np.linalg.svd(A)
    row = Vt[-1]
    F = np.empty((3,3), dtype=np.float64)
    for i in range(3):
        base = 3*i
        F[i,0]=row[base+0]; F[i,1]=row[base+1]; F[i,2]=row[base+2]
    F = _mat33_mul(_mat33_T(T2), _mat33_mul(F, T1))
    F = _rank2_project(F)
    d = F[2,2]
    if abs(d) > 1e-8: F = F/(d + 1e-8)
    return F


@njit(cache=True)
def ransac_fundamental_loransac_numba(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 1.0,
    min_samples: int = 7,
    max_trials: int = 2000,
    p_success: float = 0.99,
    use_seven_point: bool = True,
    seed: int = 0,
    # --- LO params (UPDATED) ---
    lo_enabled: bool = True,
    lo_iters: int = 16,
    lo_thresh_mul: float = 0.7,     # tighter threshold (DEFAULT < 1.0)
    lo_min_inliers: int = 20,
    lo_use_nonminimal: bool = True, # NEW: use non-minimal LO subsets
    lo_sample_size: int = 32        # NEW: target size for LO subsets
):
    """
    LO-RANSAC for the fundamental matrix (Numba).
    Returns:
      F_best: (3,3) float64
      inliers_best: (N,) uint8 mask {0,1}
      best_inliers_count: int
      best_score: float (MSAC score)
      n_trials_done: int
    """
    N = src.shape[0]

    F_best = np.zeros((3,3), dtype=np.float64)
    inliers_best = np.zeros(N, dtype=np.uint8)
    best_inliers_count = -1
    best_score = 1e300
    residuals = np.empty(N, dtype=np.float64)
    tmp_mask = np.empty(N, dtype=np.uint8)
    mask_local = np.empty(N, dtype=np.uint8)  # persistent best mask during LO
    

    # idx_buf must be big enough for both minimal and LO non-minimal sampling
    idx_cap = lo_sample_size if lo_sample_size > min_samples else min_samples
    idx_buf = np.empty(idx_cap, dtype=np.int64)

    rng_state = np.int64(seed + 1)
    adaptive_cap = max_trials
    trials = 0

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)

    while trials < max_trials and trials < adaptive_cap:
        trials += 1
        # --- global minimal sample
        k = 0
        while k < min_samples:
            rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
            j = int(rng_state % np.int64(N))
            ok = True
            for t in range(k):
                if idx_buf[t] == j:
                    ok = False; break
            if ok:
                idx_buf[k] = j
                k += 1

        pts1 = src64[idx_buf[:min_samples], :]
        pts2 = dst64[idx_buf[:min_samples], :]

        # --- estimate from minimal subset
        if (_is_degenerate_minimal(pts1) or _is_degenerate_minimal(pts2)):
            trials += 1
            continue
        
        if use_seven_point and min_samples == 7:
            F_cands = run_7point_numba(pts1, pts2)
            best_k, rng_state = _choose_best_7pt_candidate_stream(F_cands, src64, dst64, rng_state, 64)
            F_est = F_cands[best_k]
        else:
            F_est = run_8point_numba(pts1, pts2)

        # --- global scoring
        score, inl_count = _msac_score_and_mask(F_est, src64, dst64, threshold, residuals, tmp_mask)
        improved = (score < best_score)
        if improved:
            F_local = F_est
            # TIGHTER LO THRESHOLD (smaller than global)
            lo_thr = threshold * lo_thresh_mul
            if lo_thr <= 0.0:
                lo_thr = threshold  # guard

            score_local, inl_count_local = _msac_score_and_mask(
                F_local, src64, dst64, lo_thr, residuals, tmp_mask
            )
            # after computing LO baseline:
            # score_local, inl_count_local = _msac_score_and_mask(F_local, src64, dst64, lo_thr, residuals, tmp_mask)
            for i in range(N):  # snapshot the baseline mask as the current best LO mask
                mask_local[i] = tmp_mask[i]
            if lo_enabled and inl_count_local >= lo_min_inliers:
                # --- LO iterations
                degen_count=0
                for _ in range(lo_iters):
                    # Decide LO subset size
                    k_lo = inl_count_local
                    if lo_use_nonminimal:
                        # Use as large as allowed by lo_sample_size, but at least minimal
                        if k_lo > lo_sample_size:
                            k_lo = lo_sample_size
                        # Prefer non-minimal 8+; if not enough, fallback to 7 if allowed
                        if k_lo >= 8:
                            pass
                        elif k_lo == 7 and use_seven_point:
                            pass
                        else:
                            break  # not enough inliers for a meaningful LO step
                    else:
                        # Minimal LO: keep 7/8
                        k_lo = min_samples

                    # Draw k_lo unique inliers
                    #rng_state, got = _draw_unique_indices_from_mask(tmp_mask, k_lo, rng_state, idx_buf)
                    rng_state, got = _draw_unique_indices_from_mask(mask_local, k_lo, rng_state, idx_buf)

                    if got < k_lo:
                        break
                    
                    if degen_count > 8:
                        break
                    pts1_lo = src64[idx_buf[:k_lo], :]
                    pts2_lo = dst64[idx_buf[:k_lo], :]
                    if (_is_degenerate_minimal(pts1_lo) or _is_degenerate_minimal(pts2_lo)):
                        degen_count+=1
                        continue
                    # Estimate on LO subset: prefer 8-point for non-minimal
                    if k_lo >= 8:
                        F_try = run_8point_numba(pts1_lo, pts2_lo)
                    elif k_lo == 7 and use_seven_point:
                        Fc = run_7point_numba(pts1_lo, pts2_lo)
                        # choose candidate by median on this LO subset
                        best_k2, rng_state = _choose_best_7pt_candidate_stream(Fc, src64, dst64, rng_state, 64)
                        F_try = Fc[best_k2]
                    else:
                        break  # should not happen

                    # Score LO candidate under the SAME (tighter) LO threshold
                    score_try, inl_try = _msac_score_and_mask(F_try, src64, dst64, lo_thr, residuals, tmp_mask)
                    if score_try < score_local:
                        F_local = F_try
                        score_local = score_try
                        inl_count_local = inl_try
                        # accept new best -> promote tmp_mask to mask_local
                        for i in range(N):
                            mask_local[i] = tmp_mask[i]

                # After LO: move back to GLOBAL threshold; optional refit
                score_tmp, inl_tmp = _msac_score_and_mask(F_local, src64, dst64, threshold, residuals, tmp_mask)
                if inl_tmp >= 8:
                    M = inl_tmp
                    pts1_inl = np.empty((M, 2), dtype=np.float64)
                    pts2_inl = np.empty((M, 2), dtype=np.float64)
                    t = 0
                    for i in range(N):
                        if tmp_mask[i] == 1:
                            pts1_inl[t, 0] = src64[i, 0]; pts1_inl[t, 1] = src64[i, 1]
                            pts2_inl[t, 0] = dst64[i, 0]; pts2_inl[t, 1] = dst64[i, 1]
                            t += 1
                    F_local = run_8point_numba(pts1_inl, pts2_inl)
                    score_local, inl_count_local = _msac_score_and_mask(F_local, src64, dst64, threshold, residuals, tmp_mask)

            # Accept improvement (after LO / or just initial improvement)
            if score_local < best_score:
                best_score = score_local
                best_inliers_count = inl_count_local
                F_best[:, :] = F_local
                for i in range(N): inliers_best[i] = tmp_mask[i]
                adaptive_cap = min(max_trials, _ransac_max_trials(best_inliers_count, N, min_samples, p_success))

    # Final global refit on best inliers (8-point), if enough
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
        F_best = run_8point_numba(pts1_inl, pts2_inl)
        best_score, best_inliers_count = _msac_score_and_mask(F_best, src64, dst64, threshold, residuals, inliers_best)
    return F_best, inliers_best, best_inliers_count, best_score, trials
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
    D[0, 0] = S[0]
    D[1, 1] = S[1]
    return U @ D @ Vt

# =========================
# Normalization (Hartley)
# =========================

@njit(cache=True)
def _normalize_points_single(pts: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Hartley isotropic normalization for (N,2) points.
    Returns (pts_norm(N,2), T(3,3)).
    """
    N = pts.shape[0]
    x_mean0 = 0.0
    x_mean1 = 0.0
    for i in range(N):
        x_mean0 += pts[i, 0]
        x_mean1 += pts[i, 1]
    x_mean0 /= N
    x_mean1 /= N

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
        pts_n[i, 0] = scale * pts[i, 0] + T[0, 2]
        pts_n[i, 1] = scale * pts[i, 1] + T[1, 2]
    return pts_n, T

# =========================
# Seven-point: cubic solver
# =========================

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
    Solve a3*x^3 + a2*x^2 + a1*x + a0 = 0 for real roots.
    Writes up to 3 real roots into roots_out[0:count] (ascending).
    Returns count.
    """
    # Degenerate -> quadratic / linear
    if abs(a3) < 1e-18:
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
                roots_out[0] = r1; roots_out[1] = r2
            else:
                roots_out[0] = r2; roots_out[1] = r1
            return 2

    # Normalize to monic and depress
    b = a2 / a3
    c = a1 / a3
    d = a0 / a3
    b3 = b / 3.0
    p = c - b * b3
    q = 2.0 * b3 * b3 * b3 - b3 * c + d

    D = 0.25 * q * q + (p * p * p) / 27.0

    if D > 1e-18:  # one real
        sqrtD = np.sqrt(D)
        u = _cbrt(-0.5 * q + sqrtD)
        v = _cbrt(-0.5 * q - sqrtD)
        roots_out[0] = (u + v) - b3
        return 1
    elif D >= -1e-18:  # multiple real, D≈0
        u = _cbrt(-0.5 * q)
        y1 = 2.0 * u
        y2 = -u
        r1 = y1 - b3
        r2 = y2 - b3
        if abs(y1 - y2) < 1e-12:
            roots_out[0] = r1
            return 1
        if r1 <= r2:
            roots_out[0] = r1; roots_out[1] = r2
        else:
            roots_out[0] = r2; roots_out[1] = r1
        return 2
    else:  # three distinct real
        if abs(p) < 1e-18:
            roots_out[0] = -b3
            return 1
        t = np.sqrt(-4.0 * p / 3.0)  # 2*sqrt(-p/3)
        arg = (3.0 * q) / (p * t)
        if arg < -1.0: arg = -1.0
        elif arg > 1.0: arg = 1.0
        phi = np.arccos(arg) / 3.0
        r0 = 0.5 * t * np.cos(phi) * 2.0 - b3
        r1 = 0.5 * t * np.cos(phi - 2.0 * np.pi / 3.0) * 2.0 - b3
        r2 = 0.5 * t * np.cos(phi - 4.0 * np.pi / 3.0) * 2.0 - b3
        # sort
        if r0 > r1: r0, r1 = r1, r0
        if r1 > r2: r1, r2 = r2, r1
        if r0 > r1: r0, r1 = r1, r0
        roots_out[0] = r0; roots_out[1] = r1; roots_out[2] = r2
        return 3

@njit(cache=True)
def _build_A_7(x1: np.ndarray, y1: np.ndarray, x2: np.ndarray, y2: np.ndarray) -> np.ndarray:
    A = np.ones((7, 9), dtype=x1.dtype)
    for i in range(7):
        X1, Y1, X2, Y2 = x1[i], y1[i], x2[i], y2[i]
        A[i, 0] = X2 * X1; A[i, 1] = X2 * Y1; A[i, 2] = X2
        A[i, 3] = Y2 * X1; A[i, 4] = Y2 * Y1; A[i, 5] = Y2
        A[i, 6] = X1;      A[i, 7] = Y1
    return A

@njit(cache=True)
def _poly_coeffs_sampling_batch(f1: np.ndarray, f2: np.ndarray, coeffs: np.ndarray):
    """
    coeffs[b] <- [a3,a2,a1,a0] using determinant sampling at λ∈{0,1,-1,2}.
    f1,f2: (B,3,3); coeffs: (B,4)
    """
    B = f1.shape[0]
    for b in range(B):
        D = f1[b] - f2[b]
        g0  = np.linalg.det(f2[b])            # λ=0
        g1  = np.linalg.det(f2[b] + D)        # λ=1
        gm1 = np.linalg.det(f2[b] - D)        # λ=-1
        g2  = np.linalg.det(f2[b] + 2.0*D)    # λ=2
        c0 = _MINV_SAMPLE[0,0]*g0 + _MINV_SAMPLE[0,1]*g1 + _MINV_SAMPLE[0,2]*gm1 + _MINV_SAMPLE[0,3]*g2
        c1 = _MINV_SAMPLE[1,0]*g0 + _MINV_SAMPLE[1,1]*g1 + _MINV_SAMPLE[1,2]*gm1 + _MINV_SAMPLE[1,3]*g2
        c2 = _MINV_SAMPLE[2,0]*g0 + _MINV_SAMPLE[2,1]*g1 + _MINV_SAMPLE[2,2]*gm1 + _MINV_SAMPLE[2,3]*g2
        c3 = _MINV_SAMPLE[3,0]*g0 + _MINV_SAMPLE[3,1]*g1 + _MINV_SAMPLE[3,2]*gm1 + _MINV_SAMPLE[3,3]*g2
        coeffs[b, 0] = c0; coeffs[b, 1] = c1; coeffs[b, 2] = c2; coeffs[b, 3] = c3

# =========================
# Seven- & Eight-point solvers
# =========================

@njit(cache=True)
def run_7point_numba(points1: np.ndarray, points2: np.ndarray) -> np.ndarray:
    """
    Batched 7-point: (B,7,2) -> (B,3,3,3) candidates.
    """
    B = points1.shape[0]
    out = np.zeros((B, 3, 3, 3), dtype=np.float64)

    for b in range(B):
        p1n, T1 = _normalize_points_single(points1[b].astype(np.float64))
        p2n, T2 = _normalize_points_single(points2[b].astype(np.float64))

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

        coeffs = np.zeros((1, 4), dtype=np.float64)
        _poly_coeffs_sampling_batch(F1.reshape(1,3,3), F2.reshape(1,3,3), coeffs)
        a3, a2, a1, a0 = coeffs[0, 0], coeffs[0, 1], coeffs[0, 2], coeffs[0, 3]

        roots = np.zeros(3, dtype=np.float64)
        n_roots = _solve_cubic_real_cardano(a3, a2, a1, a0, roots)

        for k in range(3):
            lam = roots[k] if k < n_roots else (roots[0] if n_roots > 0 else 0.0)
            Fn = lam * F1 + (1.0 - lam) * F2
            G = _mat33_mul(Fn, T1)
            F = _mat33_mul(_mat33_T(T2), G)
            F = _rank2_project(F)
            # simple scale fix if safe
            d = F[2, 2]
            if abs(d) > 1e-8:
                F = F / (d + 1e-8)
            out[b, k] = F

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

@njit(cache=True)
def sampson_residuals_numba(F: np.ndarray, src: np.ndarray, dst: np.ndarray, out: np.ndarray):
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
        out[i] = abs(num) / np.sqrt(den)

@njit(cache=True)
def _msac_score_and_mask(F: np.ndarray, src: np.ndarray, dst: np.ndarray, threshold: float,
                         residuals: np.ndarray, out_mask: np.ndarray) -> (float, int):
    """
    MSAC score with squared clip; inlier if r < τ.
    score = sum( min(r^2, τ^2) )
    """
    sampson_residuals_numba(F, src, dst, residuals)
    N = src.shape[0]
    tau2 = threshold * threshold
    score = 0.0
    cnt = 0
    for i in range(N):
        r = residuals[i]
        r2 = r * r
        if r < threshold:
            out_mask[i] = 1
            cnt += 1
        else:
            out_mask[i] = 0
        score += (r2 if r2 < tau2 else tau2)
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
    if best_inliers <= 0 or total_points <= 0:
        return 10**9
    w = best_inliers / float(total_points)
    if w <= 0.0:
        return 10**9
    # w**min_samples (stable in Numba using loop)
    wm = 1.0
    for _ in range(min_samples):
        wm *= w
    if wm >= 1.0:
        return 1
    num = np.log(1.0 - p_success)
    den = np.log(1.0 - wm)
    if den == 0.0:  # extremely small (wm≈1)
        return 1
    est = int(np.ceil(num / den))
    if est < 1:
        est = 1
    if est > 10**9:
        est = 10**9
    return est

# =========================
# RANSAC & LO-RANSAC
# =========================

@njit(cache=True)
def ransac_fundamental_numba(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 1.0,
    min_samples: int = 7,
    max_trials: int = 1000,
    p_success: float = 0.99,
    use_seven_point: bool = True,
    seed: int = 0,
):
    """
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
    idx = np.empty(min_samples, dtype=np.int64)

    rng_state = np.int64(seed + 1)
    adaptive_cap = max_trials
    trials = 0

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)

    while trials < max_trials and trials < adaptive_cap:
        trials += 1

        # sample k unique indices
        k = 0
        while k < min_samples:
            rng_state = (np.int64(1103515245) * rng_state + np.int64(12345)) & np.int64(0x7FFFFFFF)
            j = int(rng_state % np.int64(N))
            ok = True
            for t in range(k):
                if idx[t] == j:
                    ok = False; break
            if ok:
                idx[k] = j
                k += 1

        pts1 = src64[idx, :]
        pts2 = dst64[idx, :]

        # model from minimal set
        if use_seven_point and min_samples == 7:
            F_cands = run_7point_numba(pts1.reshape(1,7,2), pts2.reshape(1,7,2))[0]
            # pick by median Sampson on the sample
            best_k = 0
            best_med = 1e300
            tmp = np.empty(min_samples, dtype=np.float64)
            for kk in range(3):
                sampson_residuals_numba(F_cands[kk], pts1, pts2, residuals[:min_samples])
                # selection-sort to get median (k is tiny)
                for t in range(min_samples): tmp[t] = residuals[t]
                for a in range(min_samples - 1):
                    mi = a
                    for b2 in range(a+1, min_samples):
                        if tmp[b2] < tmp[mi]:
                            mi = b2
                    if mi != a:
                        v = tmp[a]; tmp[a] = tmp[mi]; tmp[mi] = v
                med = tmp[min_samples // 2]
                if med < best_med:
                    best_med = med
                    best_k = kk
            F_est = F_cands[best_k]
        else:
            F_est = run_8point_numba(pts1, pts2)

        # score globally
        score, inl_count = _msac_score_and_mask(F_est, src64, dst64, threshold, residuals, tmp_mask)
        if score < best_score:
            best_score = score
            best_inliers_count = inl_count
            F_best[:, :] = F_est
            for i in range(N): inliers_best[i] = tmp_mask[i]
            adaptive_cap = _ransac_max_trials(best_inliers_count, N, min_samples, p_success)

    # final refit on inliers if enough
    if best_inliers_count >= max(8, min_samples):
        M = best_inliers_count
        pts1_inl = np.empty((M,2), dtype=np.float64)
        pts2_inl = np.empty((M,2), dtype=np.float64)
        t = 0
        for i in range(N):
            if inliers_best[i] == 1:
                pts1_inl[t, 0] = src64[i, 0]; pts1_inl[t, 1] = src64[i, 1]
                pts2_inl[t, 0] = dst64[i, 0]; pts2_inl[t, 1] = dst64[i, 1]
                t += 1
        F_best = run_8point_numba(pts1_inl, pts2_inl)
        best_score, best_inliers_count = _msac_score_and_mask(F_best, src64, dst64, threshold, residuals, inliers_best)

    return F_best, inliers_best, best_inliers_count, best_score, trials

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
def ransac_fundamental_loransac_numba(
    src: np.ndarray,
    dst: np.ndarray,
    threshold: float = 1.0,
    min_samples: int = 7,
    max_trials: int = 2000,
    p_success: float = 0.99,
    use_seven_point: bool = True,
    seed: int = 0,
    # Local Optimization (LO) params
    lo_enabled: bool = True,
    lo_iters: int = 16,
    lo_thresh_mul: float = 2.0,
    lo_min_inliers: int = 20,
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
    idx_buf = np.empty(min_samples, dtype=np.int64)

    rng_state = np.int64(seed + 1)
    adaptive_cap = max_trials
    trials = 0

    src64 = src.astype(np.float64)
    dst64 = dst.astype(np.float64)

    while trials < max_trials and trials < adaptive_cap:
        trials += 1
        # global minimal sample
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

        pts1 = src64[idx_buf, :]
        pts2 = dst64[idx_buf, :]

        # estimate from minimal subset
        if use_seven_point and min_samples == 7:
            F_cands = run_7point_numba(pts1.reshape(1,7,2), pts2.reshape(1,7,2))[0]
            best_k = 0
            best_med = 1e300
            med_buf = np.empty(min_samples, dtype=np.float64)
            for kk in range(3):
                sampson_residuals_numba(F_cands[kk], pts1, pts2, residuals[:min_samples])
                for t in range(min_samples): med_buf[t] = residuals[t]
                for a in range(min_samples - 1):
                    mi = a
                    for b2 in range(a + 1, min_samples):
                        if med_buf[b2] < med_buf[mi]:
                            mi = b2
                    if mi != a:
                        v = med_buf[a]; med_buf[a] = med_buf[mi]; med_buf[mi] = v
                med = med_buf[min_samples // 2]
                if med < best_med:
                    best_med = med
                    best_k = kk
            F_est = F_cands[best_k]
        else:
            F_est = run_8point_numba(pts1, pts2)

        # global scoring
        score, inl_count = _msac_score_and_mask(F_est, src64, dst64, threshold, residuals, tmp_mask)
        improved = (score < best_score)
        if improved:
            F_local = F_est
            lo_thr = threshold * lo_thresh_mul
            score_local, inl_count_local = _msac_score_and_mask(F_local, src64, dst64, lo_thr, residuals, tmp_mask)

            if lo_enabled and inl_count_local >= lo_min_inliers:
                for _ in range(lo_iters):
                    rng_state, got = _draw_unique_indices_from_mask(tmp_mask, min_samples, rng_state, idx_buf)
                    if got < min_samples:
                        break
                    pts1_lo = src64[idx_buf, :]
                    pts2_lo = dst64[idx_buf, :]
                    if use_seven_point and min_samples == 7:
                        Fc = run_7point_numba(pts1_lo.reshape(1,7,2), pts2_lo.reshape(1,7,2))[0]
                        # pick by median again
                        best_k = 0
                        best_med = 1e300
                        med_buf = np.empty(min_samples, dtype=np.float64)
                        for kk in range(3):
                            sampson_residuals_numba(Fc[kk], pts1_lo, pts2_lo, residuals[:min_samples])
                            for t in range(min_samples): med_buf[t] = residuals[t]
                            for a in range(min_samples - 1):
                                mi = a
                                for b2 in range(a + 1, min_samples):
                                    if med_buf[b2] < med_buf[mi]:
                                        mi = b2
                                if mi != a:
                                    v = med_buf[a]; med_buf[a] = med_buf[mi]; med_buf[mi] = v
                            med = med_buf[min_samples // 2]
                            if med < best_med:
                                best_med = med
                                best_k = kk
                        F_try = Fc[best_k]
                    else:
                        F_try = run_8point_numba(pts1_lo, pts2_lo)

                    score_try, inl_try = _msac_score_and_mask(F_try, src64, dst64, lo_thr, residuals, tmp_mask)
                    if score_try < score_local:
                        F_local = F_try
                        score_local = score_try
                        inl_count_local = inl_try

                # back to global threshold; optional refit
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

            if score_local < best_score:
                best_score = score_local
                best_inliers_count = inl_count_local
                F_best[:, :] = F_local
                for i in range(N): inliers_best[i] = tmp_mask[i]
                adaptive_cap = _ransac_max_trials(best_inliers_count, N, min_samples, p_success)

    # final global refit on best inliers (8-point), if enough
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


import numpy as np

# ---------- utilities ----------

def _frobenius_normalize(F: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(F)
    return F / (n + eps)

def _rank2_project_numpy(F: np.ndarray) -> np.ndarray:
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0.0
    return U @ np.diag(S) @ Vt

def _sampson_residuals_numpy(F: np.ndarray, src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Sampson residuals (pixels), vectorized, float64."""
    src_h = np.column_stack([src, np.ones(src.shape[0])])   # (N,3)
    dst_h = np.column_stack([dst, np.ones(dst.shape[0])])   # (N,3)
    Fsrc = (F @ src_h.T)                                    # (3,N)
    Ftd  = (F.T @ dst_h.T)                                  # (3,N)
    num  = np.sum(dst_h * Fsrc.T, axis=1)                   # (N,)
    den  = Fsrc[0]**2 + Fsrc[1]**2 + Ftd[0]**2 + Ftd[1]**2  # (N,)
    return np.abs(num) / np.sqrt(den + 1e-15)

def _pack(F: np.ndarray) -> np.ndarray:
    return F.reshape(-1)

def _unpack(p: np.ndarray) -> np.ndarray:
    return p.reshape(3, 3)

# ---------- LM refinement ----------

def select_tight_inliers(F, src, dst, tau=1.0, tight_mul=0.75):
    # squared Sampson residuals
    src_h = np.column_stack([src, np.ones(len(src))])
    dst_h = np.column_stack([dst, np.ones(len(dst))])
    Fsrc = (F @ src_h.T); Ftd = (F.T @ dst_h.T)
    num  = np.sum(dst_h * Fsrc.T, axis=1)
    den  = Fsrc[0]**2 + Fsrc[1]**2 + Ftd[0]**2 + Ftd[1]**2 + 1e-15
    r = np.abs(num) / np.sqrt(den)
    tight = r < (tight_mul * tau)
    return tight


# drop-in replacement: safer LM wrapper
def refine_fundamental_safe(F_init, src_inl, dst_inl,
                            max_iters=30, fd_eps=1e-7,
                            lm_lambda=1e-2, lm_up=10.0, lm_dn=1/3,
                            project_rank2_final=True):
    F = F_init.astype(np.float64).copy()

    def sampson(F, s, d):
        sh = np.c_[s, np.ones(len(s))]
        dh = np.c_[d, np.ones(len(d))]
        Fs = F @ sh.T
        Ftd = F.T @ dh.T
        num = np.sum(dh * Fs.T, axis=1)
        den = Fs[0]**2 + Fs[1]**2 + Ftd[0]**2 + Ftd[1]**2 + 1e-15
        return num / np.sqrt(den)

    def pack(F):  return F.reshape(-1)
    def unpack(p): return p.reshape(3,3)

    p = pack(F)
    r = sampson(unpack(p), src_inl, dst_inl)
    r = r  # signed ok; we square in cost
    cost = float(np.dot(r, r))
    lam = lm_lambda

    for _ in range(max_iters):
        # FD Jacobian (M x 9)
        J = np.zeros((r.size, 9), float)
        for j in range(9):
            dp = np.zeros_like(p); dp[j] = fd_eps
            rp = sampson(unpack(p + dp), src_inl, dst_inl)
            J[:, j] = (rp - r) / fd_eps

        JTJ = J.T @ J
        g = J.T @ r
        if np.linalg.norm(g, np.inf) < 1e-6:
            break

        A = JTJ + lam * np.eye(9)
        try:
            delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            delta = np.linalg.lstsq(A, -g, rcond=None)[0]

        p_try = p + delta
        r_try = sampson(unpack(p_try), src_inl, dst_inl)
        cost_try = float(np.dot(r_try, r_try))

        if cost_try < cost - 1e-9:
            p, r, cost = p_try, r_try, cost_try
            lam = max(1e-8, lam * lm_dn)
        else:
            lam = min(1e12, lam * lm_up)

    F_ref = unpack(p)
    if project_rank2_final:
        U, S, Vt = np.linalg.svd(F_ref)
        S[-1] = 0.0
        F_ref = U @ np.diag(S) @ Vt
    # simple scale fix; Sampson is homogeneous anyway
    if abs(F_ref[2,2]) > 1e-8:
        F_ref = F_ref / (F_ref[2,2] + 1e-8)
    return F_ref.astype(F_init.dtype, copy=False)


def refine_fundamental_nonlinear(
    F_init: np.ndarray,
    src_inl: np.ndarray,
    dst_inl: np.ndarray,
    *,
    max_iters: int = 50,
    lm_lambda: float = 1e-2,
    lm_lambda_up: float = 10.0,
    lm_lambda_down: float = 1/3,
    fd_eps: float = 1e-6,
    rank2_penalty_weight: float = 10.0,
    stop_grad_tol: float = 1e-6,
    stop_cost_tol: float = 1e-9,
    project_rank2_final: bool = True,
) -> (np.ndarray, float):
    """
    Refine F by minimizing sum of squared Sampson residuals + rank-2 penalty using LM.
    Returns (F_refined, final_cost).
    """
    # ensure float64
    src = np.asarray(src_inl, dtype=np.float64)
    dst = np.asarray(dst_inl, dtype=np.float64)

    # Start from Frobenius-normalized F
    F = _frobenius_normalize(np.asarray(F_init, dtype=np.float64))
    p = _pack(F)

    def eval_cost_and_residuals(pvec: np.ndarray):
        Fm = _frobenius_normalize(_unpack(pvec))
        r = _sampson_residuals_numpy(Fm, src, dst)  # (N,)
        # soft rank-2 penalty: sqrt(w) * sigma3 appended as residual
        s3 = np.linalg.svd(Fm, compute_uv=False)[2]
        if rank2_penalty_weight > 0:
            r = np.concatenate([r, [np.sqrt(rank2_penalty_weight) * s3]])
        cost = np.dot(r, r)
        return cost, r, Fm

    def fd_jacobian(pvec: np.ndarray, r0: np.ndarray):
        """Finite-diff Jacobian J (M x 9)."""
        m = r0.size
        J = np.zeros((m, 9), dtype=np.float64)
        for j in range(9):
            dp = np.zeros_like(pvec)
            dp[j] = fd_eps
            cost_p, r_p, _ = eval_cost_and_residuals(pvec + dp)
            J[:, j] = (r_p - r0) / fd_eps
        return J

    cost, r, F = eval_cost_and_residuals(p)
    lam = lm_lambda

    for it in range(max_iters):
        # Build normal equations
        J = fd_jacobian(p, r)                      # (M x 9)
        JTJ = J.T @ J
        g   = J.T @ r                              # (9,)
        g_norm = np.linalg.norm(g, ord=np.inf)

        if g_norm < stop_grad_tol:
            break

        # LM step: (JTJ + λ I) δ = -g
        A = JTJ + lam * np.eye(9)
        try:
            delta = np.linalg.solve(A, -g)
        except np.linalg.LinAlgError:
            # add a bit more damping
            A = JTJ + (lam * 10.0) * np.eye(9)
            delta = np.linalg.lstsq(A, -g, rcond=None)[0]

        p_trial = p + delta
        cost_trial, r_trial, F_trial = eval_cost_and_residuals(p_trial)

        if cost_trial < cost - stop_cost_tol:
            # accept
            p = p_trial
            F = F_trial
            cost = cost_trial
            r = r_trial
            lam = max(lm_lambda * 1e-6, lam * lm_lambda_down)
        else:
            # reject, increase damping
            lam = min(1e12, lam * lm_lambda_up)

    # final projection to rank-2 (common in practice)
    if project_rank2_final:
        F = _rank2_project_numpy(F)
        F = _frobenius_normalize(F)

    return F.astype(F_init.dtype, copy=False), float(cost)

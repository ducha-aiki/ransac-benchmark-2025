# select the data
import os 
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
import numpy as np
import h5py
import cv2
from tqdm import tqdm
from ransac_benchmark_imc.metrics import *
from ransac_benchmark_imc.io import load_h5, save_h5, get_output_dir

import argparse

import multiprocessing
import sys
from joblib import Parallel, delayed
import PIL
import time

try: 
    import pydegensac
except Exception as e:
    print ("pydegensac not found")
    pass
try:
    import poselib
except Exception as e:
    print ("poselib not found")
    pass
try:
    import torch
    import kornia.geometry as KG
except Exception as e:
    print ("kornia not found")
    pass
try:
    import pvsac
except Exception as e:
    print ("pvsac not found")
    pass
try:
    import pysuperansac
except Exception as e:
    print ("pysuperansac not found")
    pass
try:
    import pycolmap
except Exception as e:
    print ("pycolmap not found")
    pass
try:
    import pygcransac
except Exception as e:
    print ("pygcransac not found")
    pass
try:
    import pymagsac
except Exception as e:
    print ("pymagsac not found")
    pass
try:
    from skimage.measure import ransac as skransac
    from skimage.transform import FundamentalMatrixTransform
    from skimage7pt import FundamentalMatrixTransform7pt
    from skimage_numba import FundamentalMatrixTransform7pt as FundamentalMatrixTransform7pt_numba
except Exception as e:
    print ("skimage not found")
    pass
try:    
    from ransac_benchmark_imc.ransac_numba import ransac_fundamental_numba, ransac_fundamental_loransac_numba, refine_fundamental_nonlinear, refine_fundamental_safe
    from ransac_benchmark_imc.vibesac import ransac_fundamental_loransac_numba_refactored as ransac_fundamental_loransac_numba_vibe
except Exception as e:
    print ("numba not found, vibesac is not available")
    pass

SUPPORTED_METHODS = ['kornia-cpu', 'kornia-cpu-compiled', 'kornia-gpu',
                     'kornia-gpu-compiled', 'cv2f-ransac', 
                     'cv2f-magsac', 'cv2f-gc', 'cv2eimg', 'superansac',
                     'numba-new', 'numba-loransac', 'numba-loransac-refine','numba-loransac-vibesac',
                     'vibesac2', 'bansac',
                     'pyransac', 'degensac', 'sklearn-7pt', 'sklearn-8pt', 
                     'pygcransac', 'pymagsac',
                     'poselib', 'pycolmap', 'pvsac', 'sklearn-7pt-numba']

def norm_test_data(xs_initial, w1,h1,w2,h2):
    cx1 = (w1 - 1.0) * 0.5
    cy1 = (h1 - 1.0) * 0.5
    f1 = max(h1 - 1.0, w1 - 1.0)
    scale1 = 1.0 / f1

    T1 = np.zeros((3, 3,))
    T1[0, 0], T1[1, 1], T1[2, 2] = scale1, scale1, 1
    T1[0, 2], T1[1, 2] = -scale1 * cx1, -scale1 * cy1

    cx2 = (w2 - 1.0) * 0.5
    cy2 = (h2 - 1.0) * 0.5
    f2 = max(h2 - 1.0, w2 - 1.0)
    scale2 = 1.0 / f2

    T2 = np.zeros((3, 3,))
    T2[0, 0], T2[1, 1], T2[2, 2] = scale2, scale2, 1
    T2[0, 2], T2[1, 2] = -scale2 * cx2, -scale2 * cy2

    kp1 = (xs_initial[:, :2] - np.asarray([cx1, cy1])) / np.asarray([f1, f1])
    kp2 = (xs_initial[:, 2:] - np.asarray([cx2, cy2])) / np.asarray([f2, f2])

    xs = np.concatenate([kp1, kp2], axis=-1)
    return xs, T1, T2

def get_multi_resulst_compiled(ms_dict, m_dict, method, params, keys, prosac=False):
    if method not in ['kornia-gpu-compiled','kornia-gpu','kornia-cpu-compiled','kornia-cpu']:
        raise ValueError('Unknown method')
    BS = 512
    max_iter_batch = params['maxiter'] // BS 
    RR = KG.ransac.RANSAC(model_type='fundamental_7pt', inl_th = params['inl_th'], 
                          confidence = params['conf'], 
                          max_iter = max_iter_batch,
                          batch_size = BS)
    if 'compiled' in method: 
        print ("Compiling")
        RR.minimal_solver = torch.compile(RR.minimal_solver)
        RR.polisher_solver = torch.compile(RR.polisher_solver)
        RR.error_fn = torch.compile(RR.error_fn)
        # Warmup
        for ii,k in enumerate(keys):
            ms = ms_dict[k]
            m = m_dict[k]
            mask = ms <= params['match_th']
            tentatives = m[mask]
            tentative_idxs = np.arange(len(mask))[mask]
            if tentatives.shape[0] <= 10:
                out_results.append((np.eye(3), np.array([False] * len(mask)), 0))
                continue
            if 'gpu' in method:
                pts1 = torch.from_numpy(tentatives[:, :2]).view(-1, 2).float().cuda()
                pts2 = torch.from_numpy(tentatives[:, 2:]).view(-1, 2).float().cuda()
            else:
                pts1 = torch.from_numpy(tentatives[:, :2]).view(-1, 2).float()
                pts2 = torch.from_numpy(tentatives[:, 2:]).view(-1, 2).float()
            src_pts = tentatives[:, :2]
            dst_pts = tentatives[:, 2:]
            if prosac:
                scores = ms[mask]
                from_best = np.argsort(scores)
                src_pts = src_pts[from_best]
                dst_pts = dst_pts[from_best]
                tentative_idxs = tentative_idxs[from_best]
            if tentatives.shape[0] <= 10:
                out_results.append((np.eye(3), np.array([False] * len(mask)), 0))
                continue
            torch.cuda.synchronize()
            tic = time.perf_counter()
            if 'gpu' in method:
                pts1 = torch.from_numpy(src_pts).view(-1, 2).float().cuda()
                pts2 = torch.from_numpy(dst_pts).view(-1, 2).float().cuda()
            else:
                pts1 = torch.from_numpy(src_pts).view(-1, 2).float()
                pts2 = torch.from_numpy(dst_pts).view(-1, 2).float()
            F, mask_inl = RR(pts1, pts2)
            break
        print (f"Warmup done")
        #
    out_results = []
    for ii,k in enumerate(tqdm(keys)):
        ms = ms_dict[k]
        m = m_dict[k]
        mask = ms <= params['match_th']
        tentatives = m[mask]
        tentative_idxs = np.arange(len(mask))[mask]
        if tentatives.shape[0] <= 10:
            out_results.append((np.eye(3), np.array([False] * len(mask)), 0))
            continue
        if 'gpu' in method:
            pts1 = torch.from_numpy(tentatives[:, :2]).view(-1, 2).float().cuda()
            pts2 = torch.from_numpy(tentatives[:, 2:]).view(-1, 2).float().cuda()
        else:
            pts1 = torch.from_numpy(tentatives[:, :2]).view(-1, 2).float()
            pts2 = torch.from_numpy(tentatives[:, 2:]).view(-1, 2).float()
        src_pts = tentatives[:, :2]
        dst_pts = tentatives[:, 2:]
        if prosac:
            scores = ms[mask]
            from_best = np.argsort(scores)
            src_pts = src_pts[from_best]
            dst_pts = dst_pts[from_best]
            tentative_idxs = tentative_idxs[from_best]
        if tentatives.shape[0] <= 10:
            out_results.append((np.eye(3), np.array([False] * len(mask)), 0))
            continue
        torch.cuda.synchronize()
        tic = time.perf_counter()
        if 'gpu' in method:
            pts1 = torch.from_numpy(src_pts).view(-1, 2).float().cuda()
            pts2 = torch.from_numpy(dst_pts).view(-1, 2).float().cuda()
        else:
            pts1 = torch.from_numpy(src_pts).view(-1, 2).float()
            pts2 = torch.from_numpy(dst_pts).view(-1, 2).float()
        F, mask_inl = RR(pts1, pts2)
        torch.cuda.synchronize()
        toc = time.perf_counter()
        F = F.detach().cpu().numpy().reshape(3,3)
        mask_inl = mask_inl.detach().cpu().numpy().reshape(-1)>0
        final_inliers = np.array([False] * len(mask))
        if F is not None:
            for i, x in enumerate(mask_inl):
                final_inliers[tentative_idxs[i]] = x
        out_results.append((F, final_inliers, toc - tic))
    return out_results

def get_probabilities(tentatives, assumed_order=True):
    probabilities = []
    # Since the correspondences are assumed to be ordered by their SNN ratio a priori,
    # we just assign a probability according to their order.
    if assumed_order:
        arange = np.arange(len(tentatives))
        probabilities = 1.0 - arange / len(tentatives)
    else:
        probabilities = np.ones(len(tentatives)) / len(tentatives)
    return probabilities

def get_single_result(ms, m, method, params, w1 = None, h1 = None, w2 = None, h2  = None, prosac=False):
    mask = ms <= params['match_th']
    tentatives = m[mask]
    tentative_idxs = np.arange(len(mask))[mask]
    src_pts = tentatives[:, :2]
    dst_pts = tentatives[:, 2:]
    scores = ms[mask]
    if tentatives.shape[0] <= 12:
        return np.eye(3), np.array([False] * len(mask)), 0
    tic = time.perf_counter()
    if prosac:
        from_best = np.argsort(scores)
        tentatives = tentatives[from_best]
        src_pts = src_pts[from_best]
        dst_pts = dst_pts[from_best]
        tentative_idxs = tentative_idxs[from_best]
        scores = scores[from_best]
    if method == 'cv2f-ransac':
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, 
                                                cv2.RANSAC, 
                                                ransacReprojThreshold=params['inl_th'],
                                                confidence=params['conf'],
                                                maxIters=params['maxiter'])
    elif method == 'cv2f-magsac':
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, 
                                                cv2.USAC_MAGSAC, 
                                                ransacReprojThreshold=params['inl_th'],
                                                confidence=params['conf'],
                                                maxIters=params['maxiter'])
    elif method == 'bansac-ransac':
        #
        bansac_params = cv2.UsacParams()
        bansac_params.score = cv2.SCORE_METHOD_MAGSAC
        bansac_params.loMethod = cv2.LOCAL_OPTIM_INNER_AND_ITER_LO
        bansac_params.threshold = params['inl_th']
        bansac_params.confidence = params['conf']
        bansac_params.maxIterations  = params['maxiter']
        bansac_params.sampler = cv2.SAMPLING_BANSAC
        bansac_params.weights = 1 -np.array(scores)
        # BANSAC patches OpenCV, so we will use the original OpenCV function, but under bansac conda environment
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts,  bansac_params)
    elif method == 'bansac-magsac':
        bansac_params = cv2.UsacParams()
        bansac_params.score = cv2.SCORE_METHOD_MAGSAC
        bansac_params.loMethod = cv2.LOCAL_OPTIM_INNER_AND_ITER_LO
        bansac_params.threshold = params['inl_th']
        bansac_params.confidence = params['conf']
        bansac_params.maxIterations  = params['maxiter']
        bansac_params.sampler = cv2.SAMPLING_BANSAC
        bansac_params.weights = 1 -np.array(scores)
        # BANSAC patches OpenCV, so we will use the original OpenCV function, but under bansac conda environment
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts,  bansac_params)
    elif method == 'cv2f-gc':
        F, mask_inl = cv2.findFundamentalMat(src_pts, dst_pts, 
                                                cv2.USAC_ACCURATE, 
                                                ransacReprojThreshold=params['inl_th'],
                                                confidence=params['conf'],
                                                maxIters=params['maxiter'])
    elif method == 'poselib':
        F, info = poselib.estimate_fundamental(src_pts, 
                                                      dst_pts, {'max_epipolar_error': params['inl_th'], 
                                                                'progressive_sampling': prosac,
                                                                'max_iterations': params['maxiter'],
                                                                
                                                                'success_prob': params['conf'],
                                                                }, {})
        mask_inl = info['inliers']
    elif method == 'pycolmap':
        opts = pycolmap.RANSACOptions({'max_error': params['inl_th'], 
                                       'max_num_trials': params['maxiter'],
                                        'min_num_trials': min(1000, params['maxiter']),
                                        'confidence': params['conf']})
        res = pycolmap.estimate_fundamental_matrix(src_pts, dst_pts, opts)
        mask_inl = res['inlier_mask']
        F = res['F']
    elif method == 'kornia-cpu':
        BS = 512
        max_iter_batch = params['maxiter'] // BS 
        RR = KG.ransac.RANSAC(model_type='fundamental_7pt', inl_th = params['inl_th'], 
                              confidence = params['conf'], 
                              max_iter = max_iter_batch,
                              batch_size = BS)
        pts1 = torch.from_numpy(src_pts).view(-1, 2)
        pts2 = torch.from_numpy(dst_pts).view(-1, 2)
        F, mask_inl = RR(pts1.float(), pts2.float())
        F = F.detach().cpu().numpy().reshape(3,3)
        mask_inl = mask_inl.detach().cpu().numpy().reshape(-1)>0
    elif method == 'kornia-gpu':
        BS = 512
        max_iter_batch = params['maxiter'] // BS 
        RR = KG.ransac.RANSAC(model_type='fundamental_7pt', inl_th = params['inl_th'], 
                              confidence = params['conf'], 
                              max_iter = max_iter_batch,
                              batch_size = BS)
        pts1 = torch.from_numpy(src_pts).view(-1, 2).float().cuda()
        pts2 = torch.from_numpy(dst_pts).view(-1, 2).float().cuda()
        F, mask_inl = RR(pts1, pts2)
        F = F.detach().cpu().numpy().reshape(3,3)
        mask_inl = mask_inl.detach().cpu().numpy().reshape(-1)>0
    elif method == 'numba-new':
        F, mask_inl, best_inliers_count, best_score, trials = ransac_fundamental_numba(src_pts, dst_pts, 
                                                                          params['inl_th'],
                                                                          min_samples=7,
                                                                          max_trials=params['maxiter'],
                                                                          p_success=params['conf'],
                                                                          use_seven_point=True, msac=True)
    elif method == 'numba-loransac':
        F, mask_inl, best_inliers_count, best_score, trials = ransac_fundamental_loransac_numba(src_pts, dst_pts, 
                                                                          params['inl_th'],
                                                                          min_samples=7,
                                                                          max_trials=params['maxiter'],
                                                                          p_success=params['conf'])
    elif method == 'numba-loransac-vibesac':
        F, mask_inl, best_inliers_count, best_score, trials = ransac_fundamental_loransac_numba_vibe(src_pts, dst_pts, 
                                                                          params['inl_th'],
                                                                          min_samples=7,
                                                                          max_trials=params['maxiter'],
                                                                          p_success=params['conf'],
                                                                          use_prosac=prosac)
    elif method == 'kornia-gpu-compiled':
        BS = 512
        max_iter_batch = params['maxiter'] // BS 
        RR = KG.ransac.RANSAC(model_type='fundamental_7pt', inl_th = params['inl_th'], 
                              confidence = params['conf'], 
                              max_iter = max_iter_batch,
                              batch_size = BS)
        pts1 = torch.from_numpy(src_pts).view(-1, 2).float().cuda()
        pts2 = torch.from_numpy(dst_pts).view(-1, 2).float().cuda()
        F, mask_inl = RR(pts1, pts2)
        F = F.detach().cpu().numpy().reshape(3,3)
        mask_inl = mask_inl.detach().cpu().numpy().reshape(-1)>0
    elif method == 'cv2eimg':
        tent_norm, T1, T2 = norm_test_data(tentatives, w1,h1,w2,h2)
        E, mask_inl = cv2.findEssentialMat(tent_norm[:, :2], tent_norm[:, 2:], 
                                           np.eye(3), cv2.RANSAC, 
                                           threshold=params['inl_th'],
                                           prob=params['conf'])
        F = np.matmul(np.matmul(T2.T, E), T1)
    elif method  == 'pyransac':
        F, mask_inl = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 
                                                px_th=params['inl_th'],
                                                conf=params['conf'],
                                                max_iters = params['maxiter'],
                                                symmetric_error_check=False,
                                                enable_degeneracy_check=False)
    elif method  == 'pygcransac':
        w1 = int(m[:, 0].max()+10)
        h1 = int(m[:, 1].max()+10)
        w2 = int(m[:, 2].max()+10)
        h2 = int(m[:, 3].max()+10)
        probabilities = get_probabilities(tentatives, assumed_order=prosac)
        F, mask_inl = pygcransac.findFundamentalMatrix(np.ascontiguousarray(tentatives), 
                                                       h1, w1, w2, h2,
                                                       probabilities,
                                                       threshold=params['inl_th'],
                                                       conf=params['conf'],
                                                       max_iters = params['maxiter'],
                                                       min_iters = min(50, params['maxiter']))
    elif method  == 'pymagsac':
        w1 = int(m[:, 0].max()+10)
        h1 = int(m[:, 1].max()+10)
        w2 = int(m[:, 2].max()+10)
        h2 = int(m[:, 3].max()+10)
        probabilities = get_probabilities(tentatives, assumed_order=prosac)
        F, mask_inl = pymagsac.findFundamentalMatrix(np.ascontiguousarray(tentatives),
                                                       h1, w1, w2, h2,
                                                       probabilities,
                                                       sampler=4,
                                                       use_magsac_plus_plus=True,
                                                       conf=params['conf'],
                                                       max_iters = params['maxiter'],
                                                       min_iters = min(50, params['maxiter']),
                                                       sigma_th=params['inl_th'])
    elif method  == 'pvsac':
        params = pvsac.Params(pvsac.EstimationMethod.Fundamental, 
                              params['inl_th'], params['conf'], params['maxiter'],
                              pvsac.SamplingMethod.SAMPLING_PROSAC if prosac else pvsac.SamplingMethod.SAMPLING_UNIFORM,
                              pvsac.ScoreMethod.SCORE_METHOD_MSAC)
        F, mask_inl = pvsac.estimate(params, np.ascontiguousarray(src_pts.astype(np.float64)), np.ascontiguousarray(dst_pts.astype(np.float64)), None, None, None, None)
    elif method  == 'degensac':
        F, mask_inl = pydegensac.findFundamentalMatrix(src_pts, dst_pts, 
                                                params['inl_th'],
                                                conf=params['conf'],
                                                max_iters = params['maxiter'],
                                                symmetric_error_check=False,
                                                enable_degeneracy_check=True)
    elif method  == 'superansac':
        config = pysuperansac.RANSACSettings()
        config.inlier_threshold = params['inl_th']
        config.min_iterations = min(50, params['maxiter'])
        config.max_iterations = params['maxiter']
        config.confidence = params['conf']
        config.sampler = pysuperansac.SamplerType.PROSAC if prosac else pysuperansac.SamplerType.Uniform
        config.scoring = pysuperansac.ScoringType.MAGSAC
        config.local_optimization = pysuperansac.LocalOptimizationType.NestedRANSAC
        config.final_optimization = pysuperansac.LocalOptimizationType.LSQ
        config.neighborhood_settings.neighborhood_grid_density = 6
        config.neighborhood_settings.neighborhood_size = 6
        w1 = m[:, 0].max()
        h1 = m[:, 1].max()
        w2 = m[:, 2].max()
        h2 = m[:, 3].max()
        F, mask_inl, score, iterations = pysuperansac.estimateFundamentalMatrix(
            np.ascontiguousarray(np.concatenate([src_pts, dst_pts], axis=1)), 
            [w1, h1, w2, h2],
            scores,

            config = config)
    elif method  == 'sklearn-7pt':
        F, mask_inl = skransac([src_pts, dst_pts],
                    FundamentalMatrixTransform7pt,
                    min_samples=7,
                    residual_threshold=params['inl_th'],
                    max_trials=params['maxiter'],
                    stop_probability=params['conf'])
        mask_inl = mask_inl.astype(bool).flatten()
        F = F.params
    elif method  == 'sklearn-7pt-numba':
        F, mask_inl = skransac([src_pts, dst_pts],
                    FundamentalMatrixTransform7pt_numba,
                    min_samples=7,
                    residual_threshold=params['inl_th'],
                    max_trials=params['maxiter'],
                    stop_probability=params['conf'])
        mask_inl = mask_inl.astype(bool).flatten()
        F = F.params
    elif method  == 'sklearn-8pt':
        try:
            #print(src_pts.shape, dst_pts.shape)
            F, mask_inl = skransac([src_pts, dst_pts],
                        FundamentalMatrixTransform,
                        min_samples=8,
                        residual_threshold=params['inl_th'],
                        max_trials=params['maxiter'],
                        stop_probability=params['conf'])
            mask_inl = mask_inl.astype(bool).flatten()
            F = F.params
        except Exception as e:
            print ("Fail!", e)
            toc = time.perf_counter()
            return np.eye(3), np.array([False] * len(mask)), tic-toc
    elif method  == 'sklearn-numba':
        try:
            F, mask_inl = skransac_numba(src_pts, dst_pts, 8, params['inl_th'], params['maxiter'], params['conf'])
        except Exception as e:
            print ("Fail!", e)
            return np.eye(3), np.array([False] * len(mask))
    else:
        raise ValueError('Unknown method')
    toc = time.perf_counter()
    final_inliers = np.array([False] * len(mask))
    if F is not None:
        for i, x in enumerate(mask_inl):
            final_inliers[tentative_idxs[i]] = x
    else:
        F = np.eye(3)
    return F, final_inliers, toc - tic



        
def create_F_submission(IN_DIR, seq, method, params, num_cores, prosac=False):
    out_model = {}
    inls = {}
    times = {}
    matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
    matches_scores = load_h5(f'{IN_DIR}/{seq}/match_conf.h5')
    keys = [k for k in matches.keys()]
    BATCH = 32 
    PRE_DISPATCH = "3*n_jobs"

    if ('gpu' in method):
        results = get_multi_resulst_compiled(matches_scores, matches, method, params, keys, prosac)
        for i, k in enumerate(keys):
            v = results[i]
            out_model[k] = v[0]
            inls[k] = v[1]
            times[k] = v[2]
    elif ('vsac' in method):
        results =[]
        for i,k in enumerate(tqdm(keys)):
            if i>1580:
                print (f"{i=} {k=}, input:")
                print (f'{matches[k]=}')
            v = get_single_result(matches_scores[k], matches[k], method, params, prosac=prosac)
            results.append(v)
            out_model[k] = v[0]
            inls[k] = v[1]
            times[k] = v[2]
    else:
        results = Parallel(n_jobs=num_cores,
                           batch_size=BATCH,
                           backend="loky",
                           prefer="processes",
                           pre_dispatch=PRE_DISPATCH
                           )(delayed(get_single_result)(matches_scores[k], matches[k], method, params, prosac=prosac) for k in tqdm(keys))
        for i, k in enumerate(keys):
            v = results[i]
            out_model[k] = v[0]
            inls[k] = v[1]
            times[k] = v[2]
    return out_model, inls, times


def estimate_dir_split(split, method, inlier_th=0.75, conf=0.999, maxiter=100000, 
                       match_th=0.85, prosac=False, force=False, data_dir='f_data'):
    """
    Estimate fundamental matrices for a given split using the specified method.
    
    Args:
        split: Split to run on ('val' or 'test')
        method: RANSAC method to use (must be in SUPPORTED_METHODS)
        inlier_th: Inlier threshold (default: 0.75)
        conf: Confidence level (default: 0.999)
        maxiter: Maximum number of iterations (default: 100000)
        match_th: Match filtering threshold (default: 0.85)
        prosac: Use PROSAC sampling (default: False)
        force: Force recompute if results exist (default: False)
        data_dir: Path to the data directory (default: 'f_data')
        
    Returns:
        output_dir: Directory where results were saved
    """
    if split not in ['val', 'test']:
        raise ValueError('Unknown value for split. Must be "val" or "test"')
    
    if method.lower() not in SUPPORTED_METHODS:
        raise ValueError(f'Unknown value {method.lower()} for method. Must be one of {SUPPORTED_METHODS}')
    
    NUM_RUNS = 1
    if split == 'test':
        NUM_RUNS = 1
    
    params = {
        "maxiter": maxiter,
        "inl_th": inlier_th,
        "conf": conf,
        "match_th": match_th,
        "PROSAC": prosac
    }
    
    problem = 'f'
    OUT_DIR = get_output_dir(problem, split, method, params)
    IN_DIR = os.path.join(data_dir, split) 
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    try:
        num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
    except Exception as e: # macos likely
        num_cores = int(os.cpu_count() *0.9)
    if method == 'pvsac':
        num_cores = 4
    
    for run in range(NUM_RUNS):
        seqs = os.listdir(IN_DIR)
        for seq in seqs:
            print(f'Working on {seq}')
            out_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
            out_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
            out_times_fname = os.path.join(OUT_DIR, f'submission_times_seq_{seq}_run_{run}.h5')
            if os.path.isfile(out_models_fname) and not force:
                print(f"Submission file {out_models_fname} already exists, skipping")
                continue
            if 'kornia' in method:
                with torch.inference_mode():
                    models, inlier_masks, times = create_F_submission(IN_DIR, seq, method, params, num_cores, prosac)
            else:
                models, inlier_masks, times = create_F_submission(IN_DIR, seq, method, params, num_cores, prosac)
            save_h5(models, out_models_fname)
            save_h5(inlier_masks, out_inliers_fname)
            save_h5(times, out_times_fname)
    print('Done!')
    return OUT_DIR


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split",
        default='val',
        type=str,
        help='split to run on. Can be val or test') 
    parser.add_argument(
        "--method", default='cv2f-ransac', type=str,
        help=f'RANSAC method. Can be one of {SUPPORTED_METHODS}')
    parser.add_argument(
        "--inlier_th",
        default=0.75,
        type=float,
        help='inlier threshold. Default is 0.75')
    parser.add_argument(
        "--PROSAC", action='store_true',
        help='use PROSAC')
    parser.add_argument(
        "--conf",
        default=0.999,
        type=float,
        help='confidence Default is 0.999')
    parser.add_argument(
        "--maxiter",
        default=100000,
        type=int,
        help='max iter Default is 100000')
    parser.add_argument(
        "--match_th",
        default=0.85,
        type=float,
        help='match filtering th. Default is 0.85')
    parser.add_argument(
        "--force", action='store_true',
        help='force recompute if results exist')
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
    
    args = parser.parse_args()
    
    estimate_dir_split(
        split=args.split,
        method=args.method,
        inlier_th=args.inlier_th,
        conf=args.conf,
        maxiter=args.maxiter,
        match_th=args.match_th,
        prosac=args.PROSAC,
        force=args.force,
        data_dir=args.data_dir
    )

# select the data
import os
import numpy as np
import argparse

from ransac_benchmark_imc.estimation import estimate_dir_split, get_output_dir, SUPPORTED_METHODS
from ransac_benchmark_imc.evaluation import evaluate_dir_split
from ransac_benchmark_imc.io import load_h5


def tune_hyperparameters(method, conf=0.999, maxiter=2000, prosac=False, 
                         data_dir='f_data', inl_ths=None, match_ths=None,
                         test_iters = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000],
                         test_confs = [0.99, 0.999, 0.9999]):
    """
    Search for the best hyperparameters on the validation set.
    
    Args:
        method: RANSAC method to use
        conf: Confidence level (default: 0.999)
        maxiter: Maximum number of iterations (default: 100000)
        prosac: Use PROSAC sampling (default: False)
        data_dir: Path to the data directory (default: 'f_data')
        inl_ths: List of inlier thresholds to test (default: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0])
        match_ths: List of match thresholds to test (default: [0.75, 0.8, 0.85])
        
    Returns:
        best_params: Dictionary with best hyperparameters and mAA score
    """
    if inl_ths is None:
        inl_ths = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    if match_ths is None:
        match_ths = [0.75, 0.8, 0.85]
    
    print(f"Searching hypers for {method}, conf={conf}, maxIters={maxiter}")
    
    res = {}
    for m_th in match_ths:
        for inl_th in inl_ths:
            key = f'{inl_th}_{m_th}'
            print(f'inlier threshold = {inl_th}, match threshold={m_th}')
            
            # Run estimation
            estimate_dir_split(
                split='val',
                method=method,
                inlier_th=inl_th,
                conf=conf,
                maxiter=maxiter,
                match_th=m_th,
                prosac=prosac,
                force=False,
                data_dir=data_dir
            )
            
            # Get output directory
            params = {
                "maxiter": maxiter,
                "inl_th": inl_th,
                "conf": conf,
                "match_th": m_th,
                "PROSAC": prosac
            }
            OUT_DIR = get_output_dir('f', 'val', method, params)
            
            # Run evaluation
            evaluate_dir_split(
                submission_dir=OUT_DIR,
                split='val',
                data_dir=data_dir,
                num_runs=None,
                force=False
            )
            
            # Load results
            out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
            final_res = load_h5(out_maa_final_fname)
            res[key] = final_res['mAA']
    
    # Find best hyperparameters
    max_MAA = 0
    inl_good = 0
    match_good = 0
    for k, v in res.items():
        if max_MAA < v:
            max_MAA = v
            pars = k.split('_')
            match_good = float(pars[1])
            inl_good = float(pars[0])
    
    print(f"The best hyperparameters for {method}, conf={conf}, maxIters={maxiter} are")
    print(f"inlier_th = {inl_good}, snn_ratio = {match_good}. Validation mAA = {max_MAA}")
    
    # Create submission with best parameters
    print("Creating submission")
    for test_maxiter in test_iters:
        for test_conf in test_confs:
            print (f"Testing with maxiter={test_maxiter}, conf={test_conf}")
            estimate_dir_split(
                split='test',
                method=method,
                inlier_th=inl_good,
                conf=test_conf,
                maxiter=test_maxiter,
                match_th=match_good,
                prosac=prosac,
                force=False,
                data_dir=data_dir
            )
    print('Done!')
    
    return {
        'inlier_th': inl_good,
        'match_th': match_good,
        'validation_mAA': max_MAA,
        'all_results': res
    }


if __name__ == '__main__':
    # Search for the best hyperparameters on the validation set
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--method", default=['cv2f-ransac'], type=str, nargs='+',
        help=f'RANSAC method(s) (e.g., {SUPPORTED_METHODS}). Can specify multiple methods separated by spaces.')
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
    parser.add_argument(
        "--conf",
        default=0.999,
        type=float,
        help='confidence. Default is 0.999')
    parser.add_argument(
        "--maxiter",
        default=2000,
        type=int,
        help='max iter. Default is 100000')
    parser.add_argument(
        "--PROSAC", action='store_true',
        help='use PROSAC')
    
    args = parser.parse_args()
    
    # Loop through all methods
    all_results = {}
    for method in args.method:
        print(f"\n{'='*80}")
        print(f"Starting hyperparameter tuning for method: {method}")
        print(f"{'='*80}\n")
        test_iters = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
        if 'sklearn' in method:
            test_iters = [100, 200, 500, 1000, 2000, 5000, 10000]
        result = tune_hyperparameters(
            method=method,
            conf=args.conf,
            maxiter=args.maxiter,
            prosac=args.PROSAC,
            data_dir=args.data_dir,
            test_iters=test_iters,
           # test_iters=[5000, 10000, 20000, 50000, 100000],
            test_confs=[0.99, 0.999, 0.9999]
        )
        all_results[method] = result
        
        print(f"\n{'='*80}")
        print(f"Completed hyperparameter tuning for method: {method}")
        print(f"Best inlier_th: {result['inlier_th']}, Best match_th: {result['match_th']}, Validation mAA: {result['validation_mAA']}")
        print(f"{'='*80}\n")
    
    # Print summary if multiple methods
    if len(args.method) > 1:
        print(f"\n{'='*80}")
        print("SUMMARY OF ALL METHODS:")
        print(f"{'='*80}")
        for method, result in all_results.items():
            print(f"{method}: inlier_th={result['inlier_th']}, match_th={result['match_th']}, validation_mAA={result['validation_mAA']:.4f}")
        print(f"{'='*80}\n")

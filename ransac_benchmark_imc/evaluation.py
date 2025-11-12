import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from tqdm import tqdm
import numpy as np
import argparse
from joblib import Parallel, delayed

from ransac_benchmark_imc.io import load_h5, save_h5, get_output_dir, load_h5_nested
from ransac_benchmark_imc.metrics import get_E_from_F, normalize_keypoints, eval_essential_matrix, calc_mAA_FE


def eval_single_result(R1, R2, T1, T2, inl_mask, K1, K2, F_pred=None, E_pred=None):
    # Accept either F_pred or E_pred
    if E_pred is not None:
        # If Essential matrix is provided directly, use it
        E_matrix = E_pred
    elif F_pred is not None:
        # If Fundamental matrix is provided, convert it to Essential matrix
        E_matrix = get_E_from_F(F_pred, K1, K2)
    else:
        # Neither F_pred nor E_pred provided
        return 3.14
    
    dR = np.dot(R2, R1.T)
    dT = T2 - np.dot(dR, T1)
    p1n = normalize_keypoints(inl_mask[:, :2], K1)
    p2n = normalize_keypoints(inl_mask[:, 2:], K2)
    return max(eval_essential_matrix(p1n, p2n, E_matrix, dR, dT))

def evaluate_results(IN_DIR, seq, models, inliers, K1_K2_format=True, roma_dir='', matrix_type='f'):
    """
    Evaluate results for a given sequence.
    
    Args:
        IN_DIR: Input directory containing the dataset
        seq: Sequence name
        models: Dictionary of predicted models (either F or E matrices)
        inliers: Dictionary of inlier masks
        K1_K2_format: Whether to use K1_K2 format for calibration
        roma_dir: Path to ROMA directory (if applicable)
        matrix_type: Type of matrix in models ('f' for Fundamental, 'e' for Essential)
    
    Returns:
        ang_errors: Dictionary of angular errors for each image pair
    """
    ang_errors = {}
    if len(roma_dir)>0:
        #print (f'{roma_dir}/{seq}_roma/matches_roma_800.h5')
        matches = load_h5_nested(f'{roma_dir}/{seq}_roma/matches_roma_800.h5')
    else:
        matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
    if matches is None:
        return None, None
    if K1_K2_format:
        K1_K2 = load_h5(f'{IN_DIR}/{seq}/K1_K2.h5')
        R = load_h5(f'{IN_DIR}/{seq}/R.h5')
        T = load_h5(f'{IN_DIR}/{seq}/T.h5')
        K = {k.split('-')[0]: K1_K2[k][0][0] for k in K1_K2.keys()}
        for k in K1_K2.keys():
            K[k.split('-')[1]] = K1_K2[k][0][1]
    else:
        fulldir = f'{IN_DIR}/{seq}/set_100/calibration'
        fnames = [os.path.join(fulldir, x) for x in os.listdir(fulldir) if (x.endswith('.h5') and not x.startswith('.'))]
        cal_dicts = {f.split('/')[-1].split('.')[0].replace('calibration_', ''): load_h5(f) for f in fnames}
        R = {k: v['R'] for k, v in cal_dicts.items()}
        T = {k: v['T'] for k, v in cal_dicts.items()}
        K = {k: v['K'] for k, v in cal_dicts.items()}
    pred_models, inl_mask = models, inliers
    num_cores = 42
    PRE_DISPATCH = "3*n_jobs"
    if len(roma_dir)>0:
        def process_match(k, m):
            img_id1 = k.split('-')[0].replace('.jpg','')
            img_id2 = k.split('-')[1].replace('.jpg','')
            E_pred = None if matrix_type == 'F' else pred_models[k]
            F_pred = None if matrix_type == 'E' else pred_models[k]
            return k, eval_single_result(R[img_id1], R[img_id2], T[img_id1], T[img_id2],
                                            m[inl_mask[k]], K[img_id1], K[img_id2],
                                            E_pred=E_pred, F_pred=F_pred)
        
        results = Parallel(n_jobs=num_cores,
                           batch_size=len(matches.keys())//num_cores,
                           backend="loky",
                           prefer="processes",
                           pre_dispatch=PRE_DISPATCH)(
            delayed(process_match)(k, m) for k, m in tqdm(matches.items())
        )
        ang_errors = {k: err for k, err in results}
    else:
        for k, m in tqdm(matches.items()):
            img_id1 = k.split('-')[0].replace('.jpg','')
            img_id2 = k.split('-')[1].replace('.jpg','')
            E_pred = None if matrix_type == 'F' else pred_models[k]
            F_pred = None if matrix_type == 'E' else pred_models[k]
            ang_errors[k] = eval_single_result(R[img_id1], R[img_id2], T[img_id1], T[img_id2],
                                                m[inl_mask[k]], K[img_id1], K[img_id2],
                                                E_pred=E_pred, F_pred=F_pred)
    return ang_errors


def process_sequence(seq, run, IN_DIR, OUT_DIR, K1_K2_format, force, roma_dir='', matrix_type='f'):
    """
    Process a single sequence for evaluation.
    
    Args:
        seq: Sequence name
        run: Run number
        IN_DIR: Input directory containing the dataset
        OUT_DIR: Output directory for results
        K1_K2_format: Whether to use K1_K2 format for calibration
        force: Force recompute if results exist
        roma_dir: Path to ROMA directory (if applicable)
        matrix_type: Type of matrix in models ('F' for Fundamental, 'E' for Essential)
    
    Returns:
        mAA: Mean Average Accuracy for the sequence
        seq_time: Mean time for the sequence
    """
    print(f'Working on {seq}')

    if 'roma' in OUT_DIR:
        in_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_roma_run_{run}.h5')
        in_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_roma_run_{run}.h5')
        in_times_fname = os.path.join(OUT_DIR, f'submission_times_seq_{seq}_roma_run_{run}.h5')
        out_errors_fname = os.path.join(OUT_DIR, f'errors_seq_{seq}_roma_run_{run}.h5')
        out_maa_fname = os.path.join(OUT_DIR, f'maa_seq_{seq}_roma_run_{run}.h5')
    else:
        in_models_fname = os.path.join(OUT_DIR, f'submission_models_seq_{seq}_run_{run}.h5')
        in_inliers_fname = os.path.join(OUT_DIR, f'submission_inliers_seq_{seq}_run_{run}.h5')
        in_times_fname = os.path.join(OUT_DIR, f'submission_times_seq_{seq}_run_{run}.h5')
        out_errors_fname = os.path.join(OUT_DIR, f'errors_seq_{seq}_run_{run}.h5')
        out_maa_fname = os.path.join(OUT_DIR, f'maa_seq_{seq}_run_{run}.h5')
    
    if os.path.isfile(out_maa_fname) and not force:
        print(f"Submission file {out_maa_fname} already exists, skipping")
        res = load_h5(out_maa_fname)
        if 'time' in res and 'mAA' in res:
            return res['mAA'], res['time']
        else:
            print("Time or mAA not found in the submission file, recomputing")
    
    if not os.path.isfile(in_models_fname) or not os.path.isfile(in_inliers_fname):
        print(f"Submission file {in_inliers_fname} is missing, cannot evaluate, skipping")
        return None, None
    
    models = load_h5(in_models_fname)
    inlier_masks = load_h5(in_inliers_fname)
    times = load_h5(in_times_fname)
    times_arr = np.array(list(times.values()))
    
    if os.path.isfile(out_errors_fname) and not force:
        print(f"Submission file {in_inliers_fname} exists, read it")
        error = load_h5(out_errors_fname)
    else:
        error = evaluate_results(IN_DIR,  seq.replace('_roma', ''), models, inlier_masks, K1_K2_format, roma_dir=roma_dir, matrix_type=matrix_type)
    
    save_h5(error, out_errors_fname)
    mAA = calc_mAA_FE({seq: error})
    seq_time = times_arr.mean()
    print(f" mAA {seq} = {mAA[seq]:.5f}, time = {seq_time:.3f}")
    save_h5({"mAA": mAA[seq], "time": seq_time}, out_maa_fname)
    return mAA[seq], seq_time


def evaluate_dir_split(submission_dir, split, data_dir='f_data', num_runs=None, force=False, roma_dir='', matrix_type='f'):
    """
    Evaluate submissions for a given split.
    
    Args:
        submission_dir: Path to the directory containing submission files
        split: Split to run on ('val' or 'test')
        data_dir: Path to the data directory
        num_runs: Number of runs to evaluate (defaults to 1 for val, 3 for test if None)
        force: Force recompute if results exist
        roma_dir: Path to ROMA directory (if applicable)
        matrix_type: Type of matrix in models ('F' for Fundamental, 'E' for Essential)
        
    Returns:
        final_mAA: The final mean Average Accuracy across all runs
        final_times: The final mean time across all runs
    """
    if split not in ['val', 'test']:
        raise ValueError('Unknown value for split. Must be "val" or "test"')
    
    # Determine number of runs
    if num_runs is not None:
        NUM_RUNS = num_runs
    else:
        NUM_RUNS = 1
    
    K1_K2_format = True
    if split == 'test':
        K1_K2_format = False

    OUT_DIR = submission_dir
    if submission_dir.endswith('.h5'):
        return 
    out_maa_final_fname = os.path.join(OUT_DIR, f'maa_FINAL.h5')
    if os.path.isfile(out_maa_final_fname) and not force:
        print(f"Submission file {out_maa_final_fname} already exists, skipping")
        res = load_h5(out_maa_final_fname)
        if 'time' in res and 'mAA' in res:
            if np.isnan(np.array(res['time'])).any():
                print("Time is NaN, recomputing")
            else:
                print(f" mAA total = {res['mAA']:.5f} time = {res['time']:.5f}")
                return res['mAA'], res['time']
        else:
            print("Time or mAA not found in the submission file, recomputing")
    IN_DIR = os.path.join(data_dir, split)
    
    if not os.path.isdir(IN_DIR):
        IN_DIR = data_dir
    
    if not os.path.isdir(OUT_DIR):
        os.makedirs(OUT_DIR)
    try:
        num_cores = int(len(os.sched_getaffinity(0)) * 0.9)
    except Exception as e: # macos likely
        num_cores = int(os.cpu_count() *0.9)
    all_maas = []
    all_times = []
    for run in range(NUM_RUNS):
        seqs = [x for x in os.listdir(IN_DIR) if not x.startswith('.')]
        if len(roma_dir)== 0:
            # Process sequences in parallel
            results = Parallel(n_jobs=len(seqs))(
                delayed(process_sequence)(seq, run, IN_DIR, OUT_DIR, K1_K2_format, force, roma_dir='', matrix_type=matrix_type)
                for seq in seqs
            )
        else:
            results = [process_sequence(seq, run, IN_DIR, OUT_DIR, K1_K2_format, force, roma_dir=os.path.join(roma_dir, split), matrix_type=matrix_type) for seq in seqs]
        # Collect results
        for maa, seq_time in results:
            if maa is not None and seq_time is not None:
                all_maas.append(maa)
                all_times.append(seq_time)

    final_mAA = (np.array(all_maas)).mean()
    final_times = np.array(all_times).mean()
    save_h5({"mAA": final_mAA, "time":final_times}, out_maa_final_fname)
    print(f"{submission_dir} mAA total = {final_mAA:.3f} time = {final_times:.4f}")
    print('Done!')
    return final_mAA, final_times


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--submission_dir",
        required=True,
        type=str,
        help='path to the directory containing submission files')
    parser.add_argument(
        "--split",
        default='val',
        type=str,
        help='split to run on. Can be val or test')
    parser.add_argument(
        "--force", action='store_true', help='Force recompute if exists')
    parser.add_argument(
        "--data_dir",
        default='f_data',
        type=str,
        help='path to the data')
    parser.add_argument(
        "--num_runs",
        default=None,
        type=int,
        help='number of runs to evaluate. If not specified, defaults to 1 for val and 3 for test')
    parser.add_argument(
        "--roma_dir",
        default='',
        type=str,
        help='path to the ROMA directory')
    parser.add_argument(
        "--matrix_type",
        default='f',
        type=str,
        choices=['f', 'e'],
        help='type of matrix in model files: f for Fundamental matrix, e for Essential matrix (default: f)')
    
    args = parser.parse_args()
    
    evaluate_dir_split(
        submission_dir=args.submission_dir,
        split=args.split,
        data_dir=args.data_dir,
        num_runs=args.num_runs,
        force=args.force,
        roma_dir=args.roma_dir,
        matrix_type=args.matrix_type
    )
        




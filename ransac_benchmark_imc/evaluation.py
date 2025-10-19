import os
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"
from tqdm import tqdm
import numpy as np
import argparse
from joblib import Parallel, delayed

from ransac_benchmark_imc.io import load_h5, save_h5, get_output_dir
from ransac_benchmark_imc.metrics import get_E_from_F, normalize_keypoints, eval_essential_matrix, calc_mAA_FE


def eval_single_result(R1, R2, T1, T2, F_pred, inl_mask, K1, K2):
    if F_pred is None:
        return 3.14
    E_cv_from_F = get_E_from_F(F_pred, K1, K2)
    dR = np.dot(R2, R1.T)
    dT = T2 - np.dot(dR, T1)
    p1n = normalize_keypoints(inl_mask[:, :2], K1)
    p2n = normalize_keypoints(inl_mask[:, 2:], K2)
    return max(eval_essential_matrix(p1n, p2n, E_cv_from_F, dR, dT))

def evaluate_results(IN_DIR, seq, models, inliers, K1_K2_format=True):
    ang_errors = {}
    matches = load_h5(f'{IN_DIR}/{seq}/matches.h5')
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
    F_pred, inl_mask = models, inliers
    for k, m in tqdm(matches.items()):
        img_id1 = k.split('-')[0]
        img_id2 = k.split('-')[1]
        ang_errors[k] = eval_single_result(R[img_id1], R[img_id2], T[img_id1], T[img_id2], F_pred[k],
                                           m[inl_mask[k]], K[img_id1], K[img_id2])
    return ang_errors


def process_sequence(seq, run, IN_DIR, OUT_DIR, K1_K2_format, force):
    """Process a single sequence for evaluation."""
    print(f'Working on {seq}')
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
        error = evaluate_results(IN_DIR, seq, models, inlier_masks, K1_K2_format)
    
    save_h5(error, out_errors_fname)
    mAA = calc_mAA_FE({seq: error})
    seq_time = times_arr.mean()
    print(f" mAA {seq} = {mAA[seq]:.5f}, time = {seq_time:.3f}")
    save_h5({"mAA": mAA[seq], "time": seq_time}, out_maa_fname)
    
    return mAA[seq], seq_time


def evaluate_dir_split(submission_dir, split, data_dir='f_data', num_runs=None, force=False):
    """
    Evaluate submissions for a given split.
    
    Args:
        submission_dir: Path to the directory containing submission files
        split: Split to run on ('val' or 'test')
        data_dir: Path to the data directory
        num_runs: Number of runs to evaluate (defaults to 1 for val, 3 for test if None)
        force: Force recompute if results exist
        
    Returns:
        final_mAA: The final mean Average Accuracy across all runs
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
        # Process sequences in parallel
        results = Parallel(n_jobs=min(num_cores, len(seqs)))(
            delayed(process_sequence)(seq, run, IN_DIR, OUT_DIR, K1_K2_format, force)
            for seq in seqs
        )
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
    
    args = parser.parse_args()
    
    evaluate_dir_split(
        submission_dir=args.submission_dir,
        split=args.split,
        data_dir=args.data_dir,
        num_runs=args.num_runs,
        force=args.force
    )
        




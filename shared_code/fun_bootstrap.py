import numpy as np
import time
from joblib import Parallel, delayed
from tqdm import tqdm

def _bootstrap_job(data_true, q_values, seed=None):
    if seed is not None:
        np.random.seed(seed)
    n = len(data_true)
    indices = np.random.randint(0, n, n)
    sampled = np.take(data_true, indices)
    return np.quantile(sampled, q_values)

def bootstrap_permutation_joblib(data_true, q_range, replicas=1000, n_jobs=-1, verbose=1):
    start = time.time()
    q_values = np.asarray(q_range)

    seeds = np.random.randint(0, 1_000_000, size=replicas)
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(
        delayed(_bootstrap_job)(data_true, q_values, seed) for seed in tqdm(seeds)
    )

    # Stack results: shape (replicas, len(q_range)) → transpose
    quantiles = np.stack(results, axis=0).T  # shape: (len(q_range), replicas)

    # Compute CI across replicas
    low_q, high_q = np.quantile(quantiles, [0.025, 0.975], axis=1)

    stop = time.time()
    print(f'Joblib bootstrap time: {round(stop - start, 2)} seconds')

    return low_q, high_q


def handler_bootstrap_permutation(wp_type, q_range, replicas=10, n_jobs=-1, bootstrap_fn=bootstrap_permutation_joblib):
    n_type = np.array(wp_type).shape[0]
    aux_qq_data = []
    for wp_ in tqdm(wp_type):
        n = wp_.shape[0]
        wp_boot = np.array([
            bootstrap_fn(wp_[xx], q_range, replicas, n_jobs=n_jobs, verbose=0) 
            for xx in range(n)
            ])
        aux_qq_data.append(wp_boot)
    return aux_qq_data
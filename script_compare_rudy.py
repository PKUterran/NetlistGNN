import json
import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau
from typing import Tuple, List, Dict, Any

test_dataset_name_iter = ('superblue19', 900)

LOG_DIR = f'log/superblue19'
logs: List[Dict[str, Any]] = []


def printout(arr1, arr2, prefix="", log_prefix=""):
    pearsonr_rho, pearsonr_pval = pearsonr(arr1, arr2)
    spearmanr_rho, spearmanr_pval = spearmanr(arr1, arr2)
    kendalltau_rho, kendalltau_pval = kendalltau(arr1, arr2)
    mae = np.sum(np.abs(arr1 - arr2)) / len(arr1)
    delta = np.abs(arr1 - arr2)
    rmse = np.sqrt(np.sum(np.multiply(delta, delta)) / len(arr1))
    print(prefix + "pearson", pearsonr_rho, pearsonr_pval)
    print(prefix + "spearman", spearmanr_rho, spearmanr_pval)
    print(prefix + "kendall", kendalltau_rho, kendalltau_pval)
    print(prefix + "MAE", mae)
    print(prefix + "RMSE", rmse)
    logs[-1].update({
        f'{log_prefix}pearson_rho': pearsonr_rho,
        f'{log_prefix}pearsonr_pval': pearsonr_pval,
        f'{log_prefix}spearmanr_rho': spearmanr_rho,
        f'{log_prefix}spearmanr_pval': spearmanr_pval,
        f'{log_prefix}kendalltau_rho': kendalltau_rho,
        f'{log_prefix}kendalltau_pval': kendalltau_pval,
        f'{log_prefix}mae': mae,
        f'{log_prefix}rmse': rmse,
    })


def load_rudy_nctu(raw_dir_name: str, given_iter: int) -> Tuple[np.ndarray, np.ndarray]:
    h_bad_cmap = np.load(f'data/{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_bad_cmap = np.load(f'data/{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    h_cmap = np.load(f'data/{raw_dir_name}/iter_{given_iter}_cmap_h.npy')
    v_cmap = np.load(f'data/{raw_dir_name}/iter_{given_iter}_cmap_v.npy')
    return h_bad_cmap + v_bad_cmap, h_cmap + v_cmap


rudy_output, nctu_output = load_rudy_nctu(test_dataset_name_iter[0], test_dataset_name_iter[1])
rudy_output, nctu_output = rudy_output.flatten(), nctu_output.flatten()

printout(rudy_output, nctu_output, "\t\tGRID_NO_INDEX: ", f'test_grid_no_index_')
with open(f'{LOG_DIR}/RUDY.json', 'w+') as fp:
    json.dump(logs, fp)

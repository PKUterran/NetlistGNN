import numpy as np
import os
import json
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


def plt_tendency(logs: List[Dict[str, Any]], fig_path: str) -> Dict[str, Any]:
    list_epoch = [log['epoch'] for log in logs]
    list_train_time = [log['train_time'] for log in logs]
    list_eval_time = [log['eval_time'] for log in logs]
    list_train_grid_no_index_pearson_rho = [log['train_grid_no_index_pearson_rho'] for log in logs]
    list_train_grid_no_index_spearmanr_rho = [log['train_grid_no_index_spearmanr_rho'] for log in logs]
    list_train_grid_no_index_kendalltau_rho = [log['train_grid_no_index_kendalltau_rho'] for log in logs]
    # list_train_grid_no_index_mae = [log['train_grid_no_index_mae'] for log in logs]
    # list_train_grid_no_index_rmse = [log['train_grid_no_index_rmse'] for log in logs]
    list_test_grid_no_index_pearson_rho = [log['test_grid_no_index_pearson_rho'] for log in logs]
    list_test_grid_no_index_spearmanr_rho = [log['test_grid_no_index_spearmanr_rho'] for log in logs]
    list_test_grid_no_index_kendalltau_rho = [log['test_grid_no_index_kendalltau_rho'] for log in logs]
    # list_test_grid_no_index_mae = [log['test_grid_no_index_mae'] for log in logs]
    # list_test_grid_no_index_rmse = [log['test_grid_no_index_rmse'] for log in logs]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(list_epoch, list_train_grid_no_index_pearson_rho, color='red', linestyle='--')
    plt.plot(list_epoch, list_train_grid_no_index_spearmanr_rho, color='green', linestyle='--')
    plt.plot(list_epoch, list_train_grid_no_index_kendalltau_rho, color='blue', linestyle='--')
    plt.plot(list_epoch, list_test_grid_no_index_pearson_rho, color='red', label='pearson')
    plt.plot(list_epoch, list_test_grid_no_index_spearmanr_rho, color='green', label='spearmanr')
    plt.plot(list_epoch, list_test_grid_no_index_kendalltau_rho, color='blue', label='kendalltau')
    plt.legend()
    plt.savefig(fig_path)

    return {
        'train_time': list_train_time[-1] / 5,
        'pearson (grid no index)': list_test_grid_no_index_pearson_rho[-1],
        'spearmanr (grid no index)': list_test_grid_no_index_spearmanr_rho[-1],
        'kendalltau (grid no index)': list_test_grid_no_index_kendalltau_rho[-1],
    }


PLT_TUPLES = [
    ('GanRoute', 'superblue19/GanRoute.json'),
]

if __name__ == '__main__':
    if not os.path.isdir('figures'):
        os.mkdir('figures')
    for name, path in PLT_TUPLES:
        try:
            with open(path) as fp:
                d = json.load(fp)
            ret = plt_tendency(d, f'figures/{name}.png')
            print(f'For {name}:')
            for k, v in ret.items():
                print(f'\t{k}: {v:.3f}')
        except FileNotFoundError:
            print(f'For {name}: not found')

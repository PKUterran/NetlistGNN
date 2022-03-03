import numpy as np
import os
import json
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


def plt_tendency(logs: List[Dict[str, Any]], fig_path: str) -> Dict[str, Any]:
    list_epoch = [log['epoch'] for log in logs]
    list_train_time = [log['train_time'] for log in logs]
    list_eval_time = [log['eval_time'] for log in logs]
    list_train_node_level_pearson_rho = [log['train_node_level_pearson_rho'] for log in logs]
    list_train_node_level_spearmanr_rho = [log['train_node_level_spearmanr_rho'] for log in logs]
    list_train_node_level_kendalltau_rho = [log['train_node_level_kendalltau_rho'] for log in logs]
    # list_train_node_level_mae = [log['train_node_level_mae'] for log in logs]
    # list_train_node_level_rmse = [log['train_node_level_rmse'] for log in logs]
    list_test_node_level_pearson_rho = [log['test_node_level_pearson_rho'] for log in logs]
    list_test_node_level_spearmanr_rho = [log['test_node_level_spearmanr_rho'] for log in logs]
    list_test_node_level_kendalltau_rho = [log['test_node_level_kendalltau_rho'] for log in logs]
    # list_test_node_level_mae = [log['test_node_level_mae'] for log in logs]
    # list_test_node_level_rmse = [log['test_node_level_rmse'] for log in logs]
    list_test_grid_no_index_pearson_rho = [log['test_grid_no_index_pearson_rho'] for log in logs]
    list_test_grid_no_index_spearmanr_rho = [log['test_grid_no_index_spearmanr_rho'] for log in logs]
    list_test_grid_no_index_kendalltau_rho = [log['test_grid_no_index_kendalltau_rho'] for log in logs]
    # list_test_grid_no_index_mae = [log['test_grid_no_index_mae'] for log in logs]
    # list_test_grid_no_index_rmse = [log['test_grid_no_index_rmse'] for log in logs]
    list_test_grid_index_pearson_rho = [log['test_grid_index_pearson_rho'] for log in logs]
    list_test_grid_index_spearmanr_rho = [log['test_grid_index_spearmanr_rho'] for log in logs]
    list_test_grid_index_kendalltau_rho = [log['test_grid_index_kendalltau_rho'] for log in logs]
    # list_test_grid_index_mae = [log['test_grid_index_mae'] for log in logs]
    # list_test_grid_index_rmse = [log['test_grid_index_rmse'] for log in logs]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(list_epoch, list_train_node_level_pearson_rho, color='red', linestyle='--')
    plt.plot(list_epoch, list_train_node_level_spearmanr_rho, color='green', linestyle='--')
    plt.plot(list_epoch, list_train_node_level_kendalltau_rho, color='blue', linestyle='--')
    plt.plot(list_epoch, list_test_node_level_pearson_rho, color='red', label='pearson')
    plt.plot(list_epoch, list_test_node_level_spearmanr_rho, color='green', label='spearmanr')
    plt.plot(list_epoch, list_test_node_level_kendalltau_rho, color='blue', label='kendalltau')
    plt.plot(list_epoch, list_test_grid_no_index_pearson_rho, color='red', linestyle='-.')
    plt.plot(list_epoch, list_test_grid_no_index_spearmanr_rho, color='green', linestyle='-.')
    plt.plot(list_epoch, list_test_grid_no_index_kendalltau_rho, color='blue', linestyle='-.')
    plt.plot(list_epoch, list_test_grid_index_pearson_rho, color='red', linestyle=':')
    plt.plot(list_epoch, list_test_grid_index_spearmanr_rho, color='green', linestyle=':')
    plt.plot(list_epoch, list_test_grid_index_kendalltau_rho, color='blue', linestyle=':')
    plt.legend()
    plt.savefig(fig_path)

    return {
        'train_time': list_train_time[-1] / 5,
        # 'train_pearson': list_train_node_level_pearson_rho[-1],
        'pearson': list_test_node_level_pearson_rho[-1],
        'spearmanr': list_test_node_level_spearmanr_rho[-1],
        'kendalltau': list_test_node_level_kendalltau_rho[-1],
        # 'train_mae': list_train_node_level_mae[-1],
        # 'mae': list_test_node_level_mae[-1],
        # 'rmse': list_test_node_level_rmse[-1],
        'pearson (grid no index)': list_test_grid_no_index_pearson_rho[-1],
        'spearmanr (grid no index)': list_test_grid_no_index_spearmanr_rho[-1],
        'kendalltau (grid no index)': list_test_grid_no_index_kendalltau_rho[-1],
        'pearson (grid index)': list_test_grid_index_pearson_rho[-1],
        'spearmanr (grid index)': list_test_grid_index_spearmanr_rho[-1],
        'kendalltau (grid index)': list_test_grid_index_kendalltau_rho[-1],
    }


PLT_TUPLES = [
    ('hyper', 'superblue19/hyper.json'),
    ('hyper-xbi', 'superblue19/hyper-xbi.json'),
    ('hyper-xgr', 'superblue19/hyper-xgr.json'),
    ('SAGE', 'superblue19/SAGE.json'),
    ('GCN', 'superblue19/GCN.json'),
    ('GAT', 'superblue19/GAT.json'),
    # ('SAGE-pos', 'superblue19/SAGE-pos.json'),
    # ('GCN-pos', 'superblue19/GCN-pos.json'),
    # ('GAT-pos', 'superblue19/GAT-pos.json'),
]

if __name__ == '__main__':
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

import numpy as np
import os
import json
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


def plt_tendency(logs: List[Dict[str, Any]], fig_path: str) -> Dict[str, Any]:
    list_epoch = [log['epoch'] for log in logs]
    list_train_time = [log['train_time'] for log in logs]
    list_eval_time = [log['eval_time'] for log in logs]
    list_train_pearson_rho = [log['train_pearson_rho'] for log in logs]
    list_train_spearmanr_rho = [log['train_spearmanr_rho'] for log in logs]
    list_train_kendalltau_rho = [log['train_kendalltau_rho'] for log in logs]
    list_train_mae = [log['train_mae'] for log in logs]
    list_train_rmse = [log['train_rmse'] for log in logs]
    list_validate_pearson_rho = [log['validate_pearson_rho'] for log in logs]
    list_validate_spearman_rho = [log['validate_spearman_rho'] for log in logs]
    list_validate_kendall_rho = [log['validate_kendall_rho'] for log in logs]
    list_test_pearson_rho = [log['test_pearson_rho'] for log in logs]
    list_test_spearmanr_rho = [log['test_spearmanr_rho'] for log in logs]
    list_test_kendalltau_rho = [log['test_kendalltau_rho'] for log in logs]
    list_test_mae = [log['test_mae'] for log in logs]
    list_test_rmse = [log['test_rmse'] for log in logs]

    fig = plt.figure(figsize=(6, 4))
    plt.plot(list_epoch, list_test_pearson_rho, color='red', label='pearson')
    plt.plot(list_epoch, list_test_spearmanr_rho, color='green', label='spearmanr')
    plt.plot(list_epoch, list_test_kendalltau_rho, color='blue', label='kendalltau')
    plt.plot(list_epoch, list_test_mae, color='purple', linestyle='--', label='mae')
    plt.plot(list_epoch, list_test_rmse, color='pink', linestyle='--', label='rmse')
    plt.legend()
    if not os.path.isdir('hpwl-figures'):
        os.mkdir('hpwl-figures')
    plt.savefig(fig_path)

    list_total = np.array(list_validate_pearson_rho) + np.array(list_validate_spearman_rho) + np.array(
        list_validate_kendall_rho)
    best_epoch = np.argmax(list_total)

    return {
        'train_time': list_train_time[best_epoch] / 5,
        'pearson': list_test_pearson_rho[best_epoch],
        'spearmanr': list_test_spearmanr_rho[best_epoch],
        'kendalltau': list_test_kendalltau_rho[best_epoch],
        'mae': list_test_mae[best_epoch],
        'rmse': list_test_rmse[best_epoch],
    }


PLT_TUPLES = [
    ('MLP', 'hpwl-superblue19/MLP.json'),
    ('Net2f', 'hpwl-superblue19/Net2f.json'),
    ('Net2a', 'hpwl-superblue19/Net2a.json'),
    ('LHNN', 'hpwl-superblue19/LHNN.json'),
    ('Ours', 'hpwl-superblue19/hyper.json'),
]

if __name__ == '__main__':
    for name, path in PLT_TUPLES:
        try:
            with open(path) as fp:
                d = json.load(fp)
            ret = plt_tendency(d, f'hpwl-figures/{name}.png')
            print(f'For {name}:')
            for k, v in ret.items():
                print(f'\t{k}: {v:.3f}')
        except FileNotFoundError:
            print(f'For {name}: not found')

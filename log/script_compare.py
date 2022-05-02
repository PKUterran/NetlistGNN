import numpy as np
import os
import json
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


COLOR_SET = ['red', 'green', 'blue', 'purple', 'orange', 'black']


def get_tendency(logs: List[Dict[str, Any]]) ->  Tuple[Dict[str, Any], int]:
    list_epoch = [log['epoch'] for log in logs]
    list_train_time = [log['train_time'] / 5 for log in logs]
    list_eval_time = [log['eval_time'] for log in logs]
    try:
        # list_train_node_level_mae = [log['train_node_level_mae'] for log in logs]
        list_train_node_level_rmse = [log['train_node_level_rmse'] for log in logs]
        list_test_grid_index_rmse = [log['test_grid_index_rmse'] for log in logs]
        list_train_node_level_pearson_rho = [log['train_node_level_pearson_rho'] for log in logs]
        list_train_node_level_spearmanr_rho = [log['train_node_level_spearmanr_rho'] for log in logs]
        list_train_node_level_kendalltau_rho = [log['train_node_level_kendalltau_rho'] for log in logs]
        list_test_node_level_pearson_rho = [log['test_node_level_pearson_rho'] for log in logs]
        list_test_node_level_spearmanr_rho = [log['test_node_level_spearmanr_rho'] for log in logs]
        list_test_node_level_kendalltau_rho = [log['test_node_level_kendalltau_rho'] for log in logs]
    except KeyError:
        list_train_node_level_rmse = [log['train_grid_index_rmse'] for log in logs]
        list_test_grid_index_rmse = [log['test_grid_index_rmse'] for log in logs]
        list_train_node_level_pearson_rho = []
        list_train_node_level_spearmanr_rho = []
        list_train_node_level_kendalltau_rho = []
        list_test_node_level_pearson_rho = []
        list_test_node_level_spearmanr_rho = []
        list_test_node_level_kendalltau_rho = []
    # list_test_node_level_mae = [log['test_node_level_mae'] for log in logs]
    # list_test_node_level_rmse = [log['test_node_level_rmse'] for log in logs]
#     list_test_grid_no_index_pearson_rho = [log['test_grid_no_index_pearson_rho'] for log in logs]
#     list_test_grid_no_index_spearmanr_rho = [log['test_grid_no_index_spearmanr_rho'] for log in logs]
#     list_test_grid_no_index_kendalltau_rho = [log['test_grid_no_index_kendalltau_rho'] for log in logs]
    # list_test_grid_no_index_mae = [log['test_grid_no_index_mae'] for log in logs]
    # list_test_grid_no_index_rmse = [log['test_grid_no_index_rmse'] for log in logs]
    list_test_grid_index_pearson_rho = [log['test_grid_index_pearson_rho'] for log in logs]
    list_test_grid_index_spearmanr_rho = [log['test_grid_index_spearmanr_rho'] for log in logs]
    list_test_grid_index_kendalltau_rho = [log['test_grid_index_kendalltau_rho'] for log in logs]
    # list_test_grid_index_mae = [log['test_grid_index_mae'] for log in logs]
    # list_test_grid_index_rmse = [log['test_grid_index_rmse'] for log in logs]
    best_epoch = -1

    return {
        'time': list_train_time,
        'rmse': list_train_node_level_rmse,
        'test_rmse': list_test_grid_index_rmse,
        'pearson': list_test_node_level_pearson_rho,
        'spearmanr': list_test_node_level_spearmanr_rho,
        'kendalltau': list_test_node_level_kendalltau_rho,
#         'pearson (grid no index)': list_test_grid_no_index_pearson_rho,
#         'spearmanr (grid no index)': list_test_grid_no_index_spearmanr_rho,
#         'kendalltau (grid no index)': list_test_grid_no_index_kendalltau_rho,
        'pearson (grid index)': list_test_grid_index_pearson_rho,
        'spearmanr (grid index)': list_test_grid_index_spearmanr_rho,
        'kendalltau (grid index)': list_test_grid_index_kendalltau_rho,
    }, best_epoch


def plt_compare(name_values: Dict[str, List[float]], fig_path: str):
    fig = plt.figure(figsize=(6, 4))
    for i, (n, values) in enumerate(name_values.items()):
        if len(values) == 0:
            continue
        plt.plot(range(1, len(values)), values[1:], color=COLOR_SET[i % len(COLOR_SET)], label=n)
    plt.legend()
    plt.savefig(fig_path)


PLT_TUPLES = [
#     ('(8,10)', 'superblue19/hyper(8,10).json'),
#     ('(16,20)', 'superblue19/hyper(16,20).json'),
#     ('(32,40)', 'superblue19/hyper.json'),
#     ('(64,80)', 'superblue19/hyper(64,80).json'),
#     ('(128,160)', 'superblue19/hyper(128,160).json'),
    
    ('2', 'superblue19/hyper(cap2).json'),
    ('5', 'superblue19/hyper.json'),
    ('10', 'superblue19/hyper(cap10).json'),
    ('20', 'superblue19/hyper(cap20).json'),
#     ('Ours', 'superblue19/hyper.json'),
#     ('Ours new 1', 'superblue19/hyper-test.json'),
#     ('Ours new 2', 'superblue19/hyper-test2.json'),
#     ('Ours (o. geom.)', 'superblue19/hyper-geom.json'),
#     ('Ours (o. geom.) new 1', 'superblue19/hyper-geom-test.json'),
#     ('Ours (o. geom.) new 2', 'superblue19/hyper-geom-test2.json'),
#     ('Ours (o. geom.) CF', 'superblue19/hyper-geom-CF.json'),
#     ('Ours (o. geom.)', 'superblue16/hyper-geom.json'),
#     ('LHNN', 'superblue16/LHNN.json'),
#     ('Ours (o. topo.)', 'superblue19/hyper-topo.json'),
#     ('Ours (o. topo.) new 1', 'superblue19/hyper-topo-test.json'),
#     ('Ours (o. topo.) new 2', 'superblue19/hyper-topo-test2.json'),
    # ('SAGE', 'superblue19/SAGE.json'),
    # ('GCN', 'superblue19/GCN.json'),
#     ('GAT', 'superblue19/GAT.json'),
    # ('SAGE (w. geom.)', 'superblue19/SAGE-pos.json'),
    # ('GCN (w. geom.)', 'superblue19/GCN-pos.json'),
    # ('GAT (w. geom.)', 'superblue19/GAT-pos.json'),
    # ('CongestionNet', 'superblue19/CongestionNet.json'),
]

if __name__ == '__main__':
    ds_p = {}
    ds_s = {}
    ds_k = {}
    ds2 = {}
    ds3 = {}
    for name, path in PLT_TUPLES:
        try:
            with open(path) as fp:
                d = json.load(fp)
            ret, be = get_tendency(d)
            ds_p[name] = ret['pearson (grid index)']
            ds_s[name] = ret['spearmanr (grid index)']
            ds_k[name] = ret['kendalltau (grid index)']
#             ds[name] = ret['pearson']
            ds2[name] = ret['rmse']
            ds3[name] = ret['test_rmse']
            print(f'For {name}:')
            for k, v in ret.items():
                if len(v) == 0:
                    continue
                print(f'\t{k}: {v[be]:.3f}')
        except FileNotFoundError:
            print(f'For {name}: not found')
    plt_compare(ds_p, fig_path='figures/compare_pearson.png')
    plt_compare(ds_s, fig_path='figures/compare_spearmanr.png')
    plt_compare(ds_k, fig_path='figures/compare_kendalltau.png')
    plt_compare(ds2, fig_path='figures/compare_loss.png')
    plt_compare(ds3, fig_path='figures/compare_eval.png')

import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


def load_json(logs: List[Dict[str, Any]]) -> Dict[str, float]:
    list_epoch = [log['epoch'] for log in logs]
    list_train_time = [log['train_time'] for log in logs]
    list_test_node_level_pearson_rho = [log.setdefault('test_node_level_pearson_rho', None) for log in logs]
    list_test_node_level_spearmanr_rho = [log.setdefault('test_node_level_spearmanr_rho', None) for log in logs]
    list_test_node_level_kendalltau_rho = [log.setdefault('test_node_level_kendalltau_rho', None) for log in logs]
    list_test_grid_no_index_pearson_rho = [log.setdefault('test_grid_no_index_pearson_rho', None) for log in logs]
    list_test_grid_no_index_spearmanr_rho = [log.setdefault('test_grid_no_index_spearmanr_rho', None) for log in logs]
    list_test_grid_no_index_kendalltau_rho = [log.setdefault('test_grid_no_index_kendalltau_rho', None) for log in logs]
    list_test_grid_index_pearson_rho = [log.setdefault('test_grid_index_pearson_rho', None) for log in logs]
    list_test_grid_index_spearmanr_rho = [log.setdefault('test_grid_index_spearmanr_rho', None) for log in logs]
    list_test_grid_index_kendalltau_rho = [log.setdefault('test_grid_index_kendalltau_rho', None) for log in logs]

    return {
        'train_time': list_train_time[-1] / 5,
        'pearson': list_test_node_level_pearson_rho[-1],
        'spearmanr': list_test_node_level_spearmanr_rho[-1],
        'kendalltau': list_test_node_level_kendalltau_rho[-1],
        'pearson (grid no index)': list_test_grid_no_index_pearson_rho[-1],
        'spearmanr (grid no index)': list_test_grid_no_index_spearmanr_rho[-1],
        'kendalltau (grid no index)': list_test_grid_no_index_kendalltau_rho[-1],
        'pearson (grid index)': list_test_grid_index_pearson_rho[-1],
        'spearmanr (grid index)': list_test_grid_index_spearmanr_rho[-1],
        'kendalltau (grid index)': list_test_grid_index_kendalltau_rho[-1],
    }


PLT_TUPLES_3 = [
    ('GCN', 'superblue19/GCN.json'),
    ('GraphSAGE', 'superblue19/SAGE.json'),
    ('GAT', 'superblue19/GAT.json'),
    ('CongestionNet', 'superblue19/CongestionNet.json'),
    ('Ours (o. topo.)', 'superblue19/hyper-xgr.json'),
    ('line', ''),

    ('GanRoute', 'superblue19/GanRoute.json'),
    ('Ours (o. geom.)', 'superblue19/hyper-xbi.json'),
    ('line', ''),

    ('GAT (w. geom.)', 'superblue19/GAT-pos.json'),
    ('LHNN', 'superblue19/LHNN.json'),
    ('Ours', 'superblue19/hyper.json'),
]

if __name__ == '__main__':
    for name, path in PLT_TUPLES_3:
        if name == 'line':
            print()
        try:
            with open(path) as fp:
                d = json.load(fp)
            ret = load_json(d)
            print(f'{name}', end='')
            for _, v in ret.items():
                if v is None:
                    print(f'\t-', end='')
                else:
                    print(f'\t{v:.3f}', end='')
            print()
        except FileNotFoundError:
            print(f'{name}')

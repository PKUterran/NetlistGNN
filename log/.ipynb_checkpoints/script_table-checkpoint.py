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

    ('GAT (w. geom.)', ''),
    ('pix2pix', ''),
    ('LHNN', ''),
    ('Ours (o. geom.)', ''),
    ('Ours (small)', ''),
    ('Ours', ''),
    ('line', ''),

    ('GAT (w. geom.)', 'superblue19/GAT-pos.json'),
    ('pix2pix', 'superblue19/GanRoute.json'),
    ('LHNN', 'superblue19/LHNN.json'),
    ('Ours (o. geom.)', 'superblue19/hyper-xbi.json'),
    ('Ours (small)', 'superblue19/hyper-small.json'),
    ('Ours', 'superblue19/hyper.json'),
]

if __name__ == '__main__':
    with open('table.txt', 'w+') as fp1:
        for name, path in PLT_TUPLES_3:
            if name == 'line':
                print()
                print(file=fp1)
                continue
            try:
                with open(path) as fp:
                    d = json.load(fp)
                ret = load_json(d)
                print(f'{name}', end='')
                print(f'{name}', end='', file=fp1)
                for _, v in ret.items():
                    if v is None:
                        print(f'\t-', end='')
                        print(f'\t-', end='', file=fp1)
                    else:
                        print(f'\t{v:.3f}', end='')
                        print(f'\t{v:.3f}', end='', file=fp1)
                print()
                print(file=fp1)
            except FileNotFoundError:
                print(f'{name}')
                print(f'{name}', file=fp1)

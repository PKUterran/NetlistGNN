import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, Tuple, List, Any


def load_json(logs: List[Dict[str, Any]]) -> Tuple[Dict[str, float], int]:
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
    list_test_node_level_precision = [log.setdefault('test_node_level_precision', None) for log in logs]
    list_test_node_level_recall = [log.setdefault('test_node_level_recall', None) for log in logs]
    list_test_node_level_f1 = [log.setdefault('test_node_level_f1', None) for log in logs]
    list_test_grid_index_precision = [log.setdefault('test_grid_index_precision', None) for log in logs]
    list_test_grid_index_recall = [log.setdefault('test_grid_index_recall', None) for log in logs]
    list_test_grid_index_f1 = [log.setdefault('test_grid_index_f1', None) for log in logs]

    list_validate_grid_index_pearson_rho = [log.setdefault('validate_grid_index_pearson_rho', None) for log in logs]
    list_validate_grid_index_spearmanr_rho = [log.setdefault('validate_grid_index_spearmanr_rho', None) for log in logs]
    list_validate_grid_index_kendalltau_rho = [log.setdefault('validate_grid_index_kendalltau_rho', None) for log in logs]
    list_validate_grid_index_f1 = [log.setdefault('validate_grid_index_f1', None) for log in logs]
    if list_validate_grid_index_f1[0]:
        list_total = np.array(list_validate_grid_index_pearson_rho) +\
                     np.array(list_validate_grid_index_spearmanr_rho) +\
                     np.array(list_validate_grid_index_kendalltau_rho) +\
                     np.array(list_validate_grid_index_f1)
        best_epoch = np.argmax(list_total)
    else:
        best_epoch = -1

    return {
        'train_time': list_train_time[best_epoch] / 5,
        'pearson': list_test_node_level_pearson_rho[best_epoch],
        'spearmanr': list_test_node_level_spearmanr_rho[best_epoch],
        'kendalltau': list_test_node_level_kendalltau_rho[best_epoch],
        # 'pearson (grid no index)': list_test_grid_no_index_pearson_rho[best_epoch],
        # 'spearmanr (grid no index)': list_test_grid_no_index_spearmanr_rho[best_epoch],
        # 'kendalltau (grid no index)': list_test_grid_no_index_kendalltau_rho[best_epoch],
        'pearson (grid index)': list_test_grid_index_pearson_rho[best_epoch],
        'spearmanr (grid index)': list_test_grid_index_spearmanr_rho[best_epoch],
        'kendalltau (grid index)': list_test_grid_index_kendalltau_rho[best_epoch],
        'precision': list_test_node_level_precision[best_epoch],
        'recall': list_test_node_level_recall[best_epoch],
        'f1-score': list_test_node_level_f1[best_epoch],
        'precision (grid index)': list_test_grid_index_precision[best_epoch],
        'recall (grid index)': list_test_grid_index_recall[best_epoch],
        'f1-score (grid index)': list_test_grid_index_f1[best_epoch],
    }, best_epoch


PLT_TUPLES_3 = [
    ('GCN', 'superblue19/GCN.json'),
    ('GraphSAGE', 'superblue19/SAGE.json'),
    ('GAT', 'superblue19/GAT.json'),
    ('CongestionNet', 'superblue19/CongestionNet.json'),
    ('MPNN', 'superblue19/hyper-MPNN.json'),
    ('Ours (o. topo.)', 'superblue19/hyper-topo.json'),
    ('line', ''),

    ('GAT (w. geom.)', 'superblue19/GAT-pos.json'),
    ('Ours (o. topo. w. geom.)', 'superblue19/hyper-pos.json'),
    ('pix2pix', 'superblue19/GanRoute.json'),
    ('LHNN', 'superblue19/LHNN.json'),
    ('Ours (o. geom.)', 'superblue19/hyper-geom.json'),
#     ('Ours (small)', 'superblue19/hyper-small.json'),
    ('Ours (geom. alt. conv.)', 'superblue19/hyper-geomCF.json'),
    ('Ours', 'superblue19/hyper.json'),
    ('line', ''),
    
    ('(8,10)', 'superblue19/hyper(8,10).json'),
    ('(16,20)', 'superblue19/hyper(16,20).json'),
    ('(32,40)', 'superblue19/hyper.json'),
    ('(64,80)', 'superblue19/hyper(64,80).json'),
    ('(128,160)', 'superblue19/hyper(128,160).json'),

    ('2', 'superblue19/hyper(cap2).json'),
    ('5', 'superblue19/hyper.json'),
    ('10', 'superblue19/hyper(cap10).json'),
    ('20', 'superblue19/hyper(cap20).json'),
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
                ret, be = load_json(d)
                print(f'{name}@{be}', end='')
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

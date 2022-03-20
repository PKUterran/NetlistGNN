import os
import numpy as np
import pickle
import torch
import torch.sparse as sparse

from typing import Tuple, Dict, List

import tqdm


class SparseBinaryMatrix:
    def __init__(self, shape: Tuple[int, int]):
        self.shape = shape
        self.link_set: Dict[int, set] = {}

    def fill(self, x, y):
        self.link_set.setdefault(x, set()).add(y)

    def slice(self, xs, ys=None, use_tqdm=False):
        sbm = SparseBinaryMatrix((len(xs), len(ys) if ys is not None else self.shape[1]))
        if use_tqdm:
            t = tqdm.tqdm(enumerate(xs), total=len(xs))
        else:
            t = enumerate(xs)
        for i, x in t:
            if x not in self.link_set.keys():
                continue
            x_set = self.link_set[x]
            if ys is None:
                sbm.link_set[i] = set()
                sbm.link_set[i] |= x_set
            else:
                for j, y in enumerate(ys):
                    if y in x_set:
                        sbm.link_set.setdefault(i, set()).add(j)
        return sbm

    @property
    def y_set(self) -> set:
        ys = set()
        for _, vs in self.link_set.items():
            ys |= vs
        return ys

    def dense(self) -> torch.Tensor:
        matrix = torch.zeros(self.shape, dtype=torch.float32)
        for u, vs in self.link_set.items():
            for v in vs:
                matrix[u, v] = 1.
        return matrix


def dump_data(dir_name: str, raw_dir_name: str, given_iter, index: int, hashcode: str,
              bin_x: float = 32, bin_y: float = 40, split_size=32, force_save=False, use_tqdm=True):
    file_path = f'{dir_name}/LHNN_{given_iter}.pickle'
    if os.path.exists(file_path) and not force_save:
        return

    with open(f'{dir_name}/edge.pkl', 'rb') as fp:
        edge = pickle.load(fp)
    sizdata_x = np.load(f'{dir_name}/sizdata_x.npy')
    sizdata_y = np.load(f'{dir_name}/sizdata_y.npy')
    xdata = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    ydata = np.load(f'{dir_name}/ydata_{given_iter}.npy')
    node_pos = torch.tensor(np.vstack((xdata, ydata)), dtype=torch.float32).t()
    node_labels = np.load(f'{dir_name}/iter_{given_iter}_node_label_full_{hashcode}_.npy')
    grid_labels = np.load(f'{dir_name}/iter_{given_iter}_grid_label_full_{hashcode}_.npy')
    labels = torch.tensor(grid_labels[:, :, index].flatten(), dtype=torch.float32)
    h_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    pin_density = np.zeros_like(h_net_density)
    cell_mask = np.zeros_like(h_net_density)
    n_net = len(edge.keys())
    n_node = node_labels.shape[0]
    n_x_grid, n_y_grid = h_net_density.shape
    n_grid = n_x_grid * n_y_grid
    print(f'\t# nets: {n_net}')
    print(f'\t# nodes: {n_node}')
    print(f'\t# grids: {n_grid}')

    v_n = torch.zeros([n_net, 4], dtype=torch.float32)
    v_c = torch.zeros([n_grid, 4], dtype=torch.float32)
    a_cc = SparseBinaryMatrix((n_grid, n_grid))
    h_nc = SparseBinaryMatrix((n_grid, n_net))

    def xy_index(x_, y_):
        return x_ * n_y_grid + y_

    def xy_indexed(idx_):
        return int(idx_ / n_y_grid), idx_ % n_y_grid

    good_net_ids = []
    for i, (_, list_node_feats) in tqdm.tqdm(enumerate(edge.items()), total=len(edge.keys())):
        n_pin = len(list_node_feats)
        xs, ys = [], []
        for node, pin_px, pin_py, _ in list_node_feats:
            px, py = node_pos[node, :]
            if not px and not py:
                continue
            pin_density[int((px + pin_px) / bin_x), int((py + pin_py) / bin_y)] += 1
            xs.append(int(px / bin_x))
            ys.append(int(py / bin_y))
        if len(xs) == 0:
            span_v = span_h = 0
        else:
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            if (max_x - min_x + 1) * (max_y - min_y + 1) > 0.0025 * n_grid:
                continue
            good_net_ids.append(i)
            for xi in range(min_x, max_x + 1):
                for yi in range(min_y, max_y + 1):
                    h_nc.fill(xy_index(xi, yi), i)
            span_h = max_x - min_x + 1
            span_v = max_y - min_y + 1

        v_n[i, 0] = span_v
        v_n[i, 1] = span_h
        v_n[i, 2] = n_pin
        v_n[i, 3] = span_v * span_h
    print(f'\t# good nets: {len(good_net_ids)}')
    # v_n = v_n[good_net_ids, :]
    # h_nc = h_nc.slice(range(h_nc.shape[0]), good_net_ids, use_tqdm=True)
    # print(f'\tfinish slicing')

    for i in tqdm.tqdm(range(n_x_grid), total=n_x_grid):
        for j in range(n_y_grid):
            idx = xy_index(i, j)
            a_cc.fill(idx, idx)
            if i > 0:
                a_cc.fill(idx, xy_index(i - 1, j))
            if j > 0:
                a_cc.fill(idx, xy_index(i, j - 1))
            if i < n_x_grid - 1:
                a_cc.fill(idx, xy_index(i + 1, j))
            if j < n_y_grid - 1:
                a_cc.fill(idx, xy_index(i, j + 1))
    print(f'\tfinish a_cc fill')

    cell_mask_cnt = 0
    for i in tqdm.tqdm(range(n_node), total=n_node):
        px, py, sx, sy = xdata[i], ydata[i], sizdata_x[i], sizdata_y[i]
        min_x, max_x, min_y, max_y = \
            int((px - 1) / bin_x), int((px + sx) / bin_x), int((py - 1) / bin_y), int((py + sy) / bin_y)
        for xi in range(min_x + 1, max_x):
            for yi in range(min_y + 1, max_y):
                cell_mask[xi, yi] = 1.
                cell_mask_cnt += 1

    print(f'\t# cell masked: {cell_mask_cnt}')

    v_c[:, 0] = torch.tensor(h_net_density.flatten(), dtype=torch.float32)
    v_c[:, 1] = torch.tensor(v_net_density.flatten(), dtype=torch.float32)
    v_c[:, 2] = torch.tensor(pin_density.flatten(), dtype=torch.float32)
    v_c[:, 3] = torch.tensor(cell_mask.flatten(), dtype=torch.float32)

    c_splits = []
    x_lower = 0
    while x_lower + split_size <= n_x_grid:
        y_lower = 0
        while y_lower + split_size <= n_y_grid:
            # print(f'\t\tfrom ({x_lower}, {y_lower})')
            c_split = []
            for i in range(split_size):
                c_split.extend(range(xy_index(x_lower + i, y_lower), xy_index(x_lower + i, y_lower + split_size)))
            c_splits.append(c_split)
            y_lower += split_size
        x_lower += split_size

    list_tensors: List[Tuple[torch.Tensor, torch.Tensor, SparseBinaryMatrix, SparseBinaryMatrix, torch.Tensor]] = []
    for c_split in tqdm.tqdm(c_splits, total=len(c_splits)):
        h_nc_ = h_nc.slice(c_split)
        net_set = list(h_nc_.y_set)
        h_nc_ = h_nc_.slice(range(h_nc_.shape[0]), net_set)

        v_n_ = v_n[net_set, :]
        v_c_ = v_c[c_split, :]
        a_cc_ = a_cc.slice(c_split, c_split)
        labels_ = labels[c_split]
        list_tensors.append((v_n_, v_c_, a_cc_, h_nc_, labels_))

    with open(file_path, 'wb+') as fp:
        pickle.dump(list_tensors, fp)


def load_data(dir_name: str, given_iter
              ) -> List[Tuple[torch.Tensor, torch.Tensor, SparseBinaryMatrix, SparseBinaryMatrix, torch.Tensor]]:
    file_path = f'{dir_name}/LHNN_{given_iter}.pickle'
    with open(file_path, 'rb') as fp:
        list_tensors = pickle.load(fp)
    return [list_tensor for list_tensor in list_tensors if list_tensor[0].shape[0]]

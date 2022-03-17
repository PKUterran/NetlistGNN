import os
import numpy as np
import pickle
import torch

from typing import Tuple


def dump_data(dir_name: str, raw_dir_name: str, given_iter, index: int, hashcode: str,
              bin_x: float = 32, bin_y: float = 40, force_save=False, use_tqdm=True):
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
    labels = grid_labels[:, :, index].flatten()
    h_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    pin_density = np.zeros_like(h_net_density)
    cell_mask = np.zeros_like(h_net_density)
    n_net = len(edge.keys())
    n_node = node_labels.shape[0]
    n_x_grid, n_y_grid = h_net_density.shape
    n_grid = n_x_grid * n_y_grid

    v_n = torch.zeros([n_net, 4], dtype=torch.float32)
    v_c = torch.zeros([n_grid, 4], dtype=torch.float32)
    a_cc = torch.zeros([n_grid, n_grid], dtype=torch.float32)
    h_nc = torch.zeros([n_grid, n_net], dtype=torch.float32)

    def xy_index(x_, y_):
        return x_ * n_y_grid + y_

    def xy_indexed(idx_):
        return np.rint(idx_ / n_y_grid), idx_ % n_y_grid

    for i, (_, list_node_feats) in enumerate(edge.items()):
        n_pin = len(list_node_feats)
        xs, ys = [], []
        for node, pin_px, pin_py, _ in list_node_feats:
            px, py = node_pos[node, :]
            if not px and not py:
                continue
            pin_density[np.rint((px + pin_px) / bin_x), np.rint((py + pin_py) / bin_y)] += 1
            xs.append(np.rint(px / bin_x))
            ys.append(np.rint(py / bin_y))
        if len(xs):
            span_v = span_h = 0
        else:
            min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
            for xi in range(min_x, max_x + 1):
                for yi in range(min_y, max_y + 1):
                    h_nc[xy_index(xi, yi), i] = 1.
            span_h = max_x - min_x + 1
            span_v = max_y - min_y + 1

        v_n[i, 0] = span_v
        v_n[i, 1] = span_h
        v_n[i, 2] = n_pin
        v_n[i, 3] = span_v * span_h

    for i in range(n_x_grid):
        for j in range(n_y_grid):
            idx = xy_index(i, j)
            a_cc[idx, idx] = 1.
            if i > 0:
                a_cc[idx, xy_index(i - 1, j)] = 1.
            if j > 0:
                a_cc[idx, xy_index(i, j - 1)] = 1.
            if i < n_x_grid - 1:
                a_cc[idx, xy_index(i + 1, j)] = 1.
            if j < n_y_grid - 1:
                a_cc[idx, xy_index(i, j + 1)] = 1.

    for i in range(n_node):
        px, py, sx, sy = xdata[i], ydata[i], sizdata_x[i], sizdata_y[i]
        min_x, max_x, min_y, max_y = \
            np.rint((px - 1) / bin_x), np.rint((px + sx) / bin_x), np.rint((py - 1) / bin_y), np.rint((py + sy) / bin_y)
        for xi in range(min_x + 1, max_x):
            for yi in range(min_y + 1, max_y):
                cell_mask[xi, yi] = 1.

    v_c[:, 0] = h_net_density.flatten()
    v_c[:, 1] = v_net_density.flatten()
    v_c[:, 2] = pin_density.flatten()
    v_c[:, 3] = cell_mask.flatten()

    with open(file_path, 'wb+') as fp:
        pickle.dump((v_n, v_c, a_cc, h_nc, labels), fp)


def load_data(dir_name: str, given_iter) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    file_path = f'{dir_name}/LHNN_{given_iter}.pickle'
    with open(file_path, 'rb') as fp:
        list_tensors = pickle.load(fp)
    return list_tensors

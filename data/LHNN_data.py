import os
import numpy as np
import pickle
import torch


def dump_data(dir_name: str, raw_dir_name: str, given_iter, index: int, hashcode: str,
              bin_x: float = 32, bin_y: float = 40, force_save=False, use_tqdm=True):
    file_path = f'{dir_name}/graphs_{given_iter}.pickle'
    if os.path.exists(file_path) and not force_save:
        with open(file_path, 'rb') as fp:
            list_tuple_graph = pickle.load(fp)
        return list_tuple_graph

    with open(f'{dir_name}/edge.pkl', 'rb') as fp:
        edge = pickle.load(fp)
    sizdata_x = np.load(f'{dir_name}/sizdata_x.npy')
    sizdata_y = np.load(f'{dir_name}/sizdata_y.npy')
    pdata = np.load(f'{dir_name}/pdata.npy')
    xdata = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    ydata = np.load(f'{dir_name}/ydata_{given_iter}.npy')
    labels = np.load(f'{dir_name}/iter_{given_iter}_node_label_full_{hashcode}_.npy')
    labels = labels[:, index]
    n_node = labels.shape[0]

    node_pos = torch.tensor(np.vstack((xdata, ydata)), dtype=torch.float32).t()


def load_data():
    pass

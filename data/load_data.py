import os
import numpy as np
import pickle
import torch
import dgl
import dgl.function as fn
import tqdm

from dgl.transform import add_self_loop, metis_partition
from typing import Tuple, List, Dict


def fo_average(g):
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    g.ndata['addnlfeat'] = (g.ndata['feat']) / degrees.view(-1, 1)
    g.ndata['inter'] = torch.zeros_like(g.ndata['feat'])
    g.ndata['wts'] = torch.ones(g.number_of_nodes()) / degrees
    g.ndata['wtmsg'] = torch.zeros_like(g.ndata['wts'])
    g.update_all(message_func=fn.copy_src(src='addnlfeat', out='inter'),
                 reduce_func=fn.sum(msg='inter', out='addnlfeat'))
    g.update_all(message_func=fn.copy_src(src='wts', out='wtmsg'),
                 reduce_func=fn.sum(msg='wtmsg', out='wts'))
    hop1 = g.ndata['addnlfeat'] / (g.ndata['wts'].view(-1, 1))
    return hop1


def get_partition_list(g, p_size):
    p_gs = metis_partition(g, p_size)
    graphs = []
    for k, val in p_gs.items():
        nids = val.ndata[dgl.NID]
        nids = nids.numpy()
        graphs.append(nids)
    return graphs


def get_partition_list_random(g, p_size):
    nids = g.nodes()
    nids = np.random.permutation(nids)
    return [nids[i::p_size] for i in range(p_size)]


def node_pairs_among(nodes, max_cap=-1):
    us = []
    vs = []
    if max_cap == -1 or len(nodes) <= max_cap:
        for u in nodes:
            for v in nodes:
                if u == v:
                    continue
                us.append(u)
                vs.append(v)
    else:
        for u in nodes:
            vs_ = np.random.permutation(nodes)
            left = max_cap - 1
            for v_ in vs_:
                if left == 0:
                    break
                if u == v_:
                    continue
                us.append(u)
                vs.append(v_)
                left -= 1
    return us, vs


def get_pin_density(shape, bin_x, bin_y, xdata, ydata, edge) -> np.ndarray:
    pin_density = np.zeros(shape, dtype=np.float)
    for i, (_, list_node_feats) in tqdm.tqdm(enumerate(edge.items()), total=len(edge.keys())):
        for node, pin_px, pin_py, _ in list_node_feats:
            px, py = xdata[node], ydata[node]
            if not px and not py:
                continue
            pin_density[int((px + pin_px) / bin_x), int((py + pin_py) / bin_y)] += 1
    return pin_density


def get_node_density(shape, bin_x, bin_y, xdata, ydata) -> np.ndarray:
    node_density = np.zeros(shape, dtype=np.float)
    for x, y in zip(xdata, ydata):
        if x < 1e-5 and y < 1e-5:
            continue
        key1 = int(x / bin_x)
        key2 = int(y / bin_y)
        node_density[key1, key2] += 1
    return node_density


def feature_grid2node(grid_feature: np.ndarray, bin_x, bin_y, xdata, ydata) -> np.ndarray:
    return np.array([
        grid_feature[int(x / bin_x), int(y / bin_y)] for x, y in zip(xdata, ydata)
    ], dtype=np.float)


def load_data(dir_name: str, given_iter, index: int, hashcode: str,
              graph_scale: int = 1000, bin_x: float = 32, bin_y: float = 40, force_save=False, use_tqdm=True,
              app_name='', win_x: float = 32, win_y: float = 40, win_cap=5
              ) -> List[Tuple[dgl.DGLGraph, dgl.DGLHeteroGraph]]:
    file_path = f'{dir_name}/hetero{app_name}_{given_iter}.pickle'
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
    raw_dir_name = dir_name[:-10]
    h_net_density_grid = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_net_density_grid = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    if os.path.exists(f'{raw_dir_name}/iter_{given_iter}_net2hpwl.npy'):
        net2hpwl = np.load(f'{raw_dir_name}/iter_{given_iter}_net2hpwl.npy')
        net2hpwl[net2hpwl < 1e-4] = 1e-4
        net2hpwl = np.log10(net2hpwl)
    else:
        net2hpwl = None
    pin_density_grid = get_pin_density(h_net_density_grid.shape, bin_x, bin_y, xdata, ydata, edge)
    node_density_grid = get_node_density(h_net_density_grid.shape, bin_x, bin_y, xdata, ydata)

    labels = np.load(f'{dir_name}/iter_{given_iter}_node_label_full_{hashcode}_.npy')
    labels = labels[:, index]
    n_node = labels.shape[0]

    node_hv = torch.tensor(np.vstack((
        sizdata_x, sizdata_y, pdata,
        feature_grid2node(h_net_density_grid, bin_x, bin_y, xdata, ydata),
        feature_grid2node(v_net_density_grid, bin_x, bin_y, xdata, ydata),
        feature_grid2node(pin_density_grid, bin_x, bin_y, xdata, ydata),
        feature_grid2node(node_density_grid, bin_x, bin_y, xdata, ydata),
    )), dtype=torch.float32).t()
    node_pos = torch.tensor(np.vstack((xdata, ydata)), dtype=torch.float32).t()

    # homo_graph
    us, vs = [], []
    for net, list_node_feats in edge.items():
        nodes = [node_feats[0] for node_feats in list_node_feats]
        us_, vs_ = node_pairs_among(nodes, max_cap=8)
        us.extend(us_)
        vs.extend(vs_)
    homo_graph = add_self_loop(dgl.graph((us, vs), num_nodes=n_node))

    print(f'\t# of nodes: {n_node}')
    homo_graph.ndata['pos'] = node_pos[:n_node, :]
    homo_graph.ndata['feat'] = node_hv[:n_node, :]
    extra = fo_average(homo_graph)
    homo_graph.ndata.pop('inter')
    homo_graph.ndata.pop('addnlfeat')
    homo_graph.ndata.pop('wts')
    homo_graph.ndata.pop('wtmsg')
    homo_graph.ndata['feat'] = torch.cat([homo_graph.ndata['feat'], extra], dim=1)
    homo_graph.ndata['label'] = torch.tensor(labels[:n_node], dtype=torch.float32)
    partition_list = get_partition_list(homo_graph, int(np.ceil(n_node / graph_scale)))
    # partition_list = get_partition_list_random(homo_graph, int(np.ceil(n_node / graph_scale)))
    list_homo_graph = [dgl.node_subgraph(homo_graph, partition) for partition in partition_list]
    print('\thomo_graph generated')

    def distance_among(a: int, b: int) -> float:
        return ((xdata[a] + sizdata_x[a] * 0.5 - xdata[b] - sizdata_x[b] * 0.5) ** 2
                + (ydata[a] + sizdata_y[a] * 0.5 - ydata[b] - sizdata_y[b] * 0.5) ** 2) ** 0.5

    # hetero_graph
    n_dim = homo_graph.ndata['feat'].shape[1]
    us4, vs4 = [], []
    off_temps = [5678, 7654, 8888, 10035]
    node_pos_code = np.zeros([n_node, n_dim], dtype=np.float)
    for off_idx, (x_offset, y_offset) in enumerate([(0, 0), (win_x / 2, 0), (0, win_y / 2), (win_x / 2, win_y / 2)]):
        box_node = {}
        iter_sp = tqdm.tqdm(enumerate(zip(sizdata_x, sizdata_y, xdata, ydata)), total=n_node) \
            if use_tqdm else enumerate(zip(sizdata_x, sizdata_y, xdata, ydata))
        for i, (sx, sy, px, py) in iter_sp:
            if i >= n_node:
                continue
            if px == 0 and py == 0:
                continue
            px += x_offset
            py += y_offset
            x_1, x_2 = int(px / win_x), int((px + sx) / win_x)
            y_1, y_2 = int(py / win_y), int((py + sy) / win_y)
            for x in range(x_1, x_2 + 1):
                for y in range(y_1, y_2 + 1):
                    box_node.setdefault(f'{x}-{y}', []).append(i)
            pos_idx = 20 * (((px + sx * 0.5) / win_x) % 1.0) + 5 * (((py + sy * 0.5) / win_y) % 1.0)
            node_pos_code[i, 0::2] += np.sin(
                np.array([
                    pos_idx / (off_temps[off_idx] ** (di / n_dim)) for di in list(range(n_dim))[0::2]
                ], dtype=np.float)
            )
            node_pos_code[i, 1::2] += np.cos(
                np.array([
                    pos_idx / (off_temps[off_idx] ** ((di - 1) / n_dim)) for di in list(range(n_dim))[1::2]
                ], dtype=np.float)
            )
        #             print(pos_idx)
        #             print(node_pos_code[i])
        #             exit(123)
        us, vs = [], []
        for nodes in box_node.values():
            us_, vs_ = node_pairs_among(nodes, max_cap=win_cap)
            us.extend(us_)
            vs.extend(vs_)
        us4.extend(us)
        vs4.extend(vs)
    dis4 = [[distance_among(u, v) / 24] for u, v in zip(us4, vs4)]
    #     print(dis4)
    #     print(np.mean(dis4))
    #     print(np.std(dis4))
    #     exit(123)

    print('\thetero_graph generated 1/2')

    us, vs, he = [], [], []
    net_degree, net_label = [], []
    for net, list_node_feats in edge.items():
        net_degree.append(len(list_node_feats))
        net_label.append(net2hpwl[net] if net2hpwl is not None else 0)
        for node_feats in list_node_feats:
            us.append(node_feats[0])
            vs.append(net)
            he.append([node_feats[1], node_feats[2], node_feats[3]])
    net_hv = torch.unsqueeze(torch.tensor(net_degree, dtype=torch.float32), dim=-1)
    net_degree_ = torch.unsqueeze(torch.tensor(net_degree, dtype=torch.float32), dim=-1)
    net_label = torch.unsqueeze(torch.tensor(net_label, dtype=torch.float32), dim=-1)

    hetero_graph = dgl.heterograph({
        ('node', 'near', 'node'): (us4, vs4),
        ('node', 'pins', 'net'): (us, vs),
        ('net', 'pinned', 'node'): (vs, us),
    }, num_nodes_dict={'node': n_node, 'net': len(net_degree)})
    hetero_graph.nodes['node'].data['hv'] = homo_graph.ndata['feat']
    hetero_graph.nodes['node'].data['pos_code'] = torch.tensor(node_pos_code, dtype=torch.float32)
    hetero_graph.nodes['net'].data['hv'] = net_hv
    hetero_graph.nodes['net'].data['degree'] = net_degree_
    hetero_graph.nodes['net'].data['label'] = net_label
    # hetero_graph.edges['pins'].data['he'] = torch.tensor(he, dtype=torch.float32)
    hetero_graph.edges['pinned'].data['he'] = torch.tensor(he, dtype=torch.float32)
    hetero_graph.edges['near'].data['he'] = torch.tensor(dis4, dtype=torch.float32)

    list_hetero_graph = []
    iter_partition_list = tqdm.tqdm(partition_list, total=len(partition_list)) if use_tqdm else partition_list
    for partition in iter_partition_list:
        partition_set = set(partition)
        new_net_degree_dict = {}
        for net_id, node_id in zip(*[ns.tolist() for ns in hetero_graph.edges(etype='pinned')]):
            if node_id in partition_set:
                new_net_degree_dict.setdefault(net_id, 0)
                new_net_degree_dict[net_id] += 1
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'node': partition, 'net': list(new_net_degree_dict.keys())})
        new_net_degree = torch.unsqueeze(torch.tensor(list(new_net_degree_dict.values()), dtype=torch.float32), dim=-1)
        part_hetero_graph.nodes['net'].data['new_degree'] = new_net_degree
#         print(part_hetero_graph.nodes['net'].data['degree'])
#         print(part_hetero_graph.nodes['net'].data['new_degree'])
#         exit(123)
        list_hetero_graph.append(part_hetero_graph)
    print('\thetero_graph generated 2/2')

    list_tuple_graph = list(zip(list_homo_graph, list_hetero_graph))
    with open(file_path, 'wb+') as fp:
        pickle.dump(list_tuple_graph, fp)
    return list_tuple_graph

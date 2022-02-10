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


def load_data(dir_name: str, given_iter, index: int, hashcode: str,
              graph_scale: int = 1000, bin_x: float = 40, bin_y: float = 32, force_save=False, use_tqdm=True
              ) -> List[Tuple[dgl.DGLGraph, dgl.DGLHeteroGraph, List[dgl.DGLGraph]]]:
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

    node_hv = torch.tensor(np.vstack((sizdata_x, sizdata_y, pdata)), dtype=torch.float32).t()
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

    # grid_graph
    grid_graphs = []
    for x_offset, y_offset in [(0, 0), (bin_x / 2, 0), (0, bin_y / 2), (bin_x / 2, bin_y / 2)]:
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
            x_1, x_2 = int(px / bin_x), int((px + sx) / bin_x)
            y_1, y_2 = int(py / bin_y), int((py + sy) / bin_y)
            for x in range(x_1, x_2 + 1):
                for y in range(y_1, y_2 + 1):
                    box_node.setdefault(f'{x}-{y}', []).append(i)
        us, vs = [], []
        # print([len(nodes) for nodes in box_node.values()])
        # exit(123)
        for nodes in box_node.values():
            us_, vs_ = node_pairs_among(nodes)
            us.extend(us_)
            vs.extend(vs_)
        grid_graph = add_self_loop(dgl.graph((us, vs), num_nodes=n_node))
        grid_graphs.append(grid_graph)

    list_grid_graphs = [[dgl.node_subgraph(grid_graph, partition) for grid_graph in grid_graphs]
                        for partition in partition_list]
    print('\tgrid_graphs generated')

    # hetero_graph
    us, vs, he = [], [], []
    net_degree = []
    for net, list_node_feats in edge.items():
        net_degree.append(len(list_node_feats))
        for node_feats in list_node_feats:
            us.append(node_feats[0])
            vs.append(net)
            he.append([node_feats[1], node_feats[2], node_feats[3]])
    net_hv = torch.unsqueeze(torch.tensor(net_degree, dtype=torch.float32), dim=-1)

    hetero_graph = dgl.heterograph({
        ('node', 'pins', 'net'): (us, vs),
        ('net', 'pinned', 'node'): (vs, us),
    }, num_nodes_dict={'node': n_node, 'net': len(net_degree)})
    hetero_graph.nodes['node'].data['hv'] = homo_graph.ndata['feat']
    hetero_graph.nodes['net'].data['hv'] = net_hv
    # hetero_graph.edges['pins'].data['he'] = torch.tensor(he, dtype=torch.float32)
    hetero_graph.edges['pinned'].data['he'] = torch.tensor(he, dtype=torch.float32)

    list_hetero_graph = []
    iter_partition_list = tqdm.tqdm(partition_list, total=len(partition_list)) if use_tqdm else partition_list
    for partition in iter_partition_list:
        partition_set = set(partition)
        keep_net_ids = set()
        for net_id, node_id in zip(*[ns.tolist() for ns in hetero_graph.edges(etype='pinned')]):
            if node_id in partition_set:
                keep_net_ids.add(net_id)
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'node': partition, 'net': list(keep_net_ids)})
        list_hetero_graph.append(part_hetero_graph)
    print('\thetero_graph generated')

    list_tuple_graph = list(zip(list_homo_graph, list_hetero_graph, list_grid_graphs))
    with open(file_path, 'wb+') as fp:
        pickle.dump(list_tuple_graph, fp)
    return list_tuple_graph

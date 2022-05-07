import os
import pickle
import torch
import numpy as np
import dgl
import tqdm
from typing import List, Tuple

from data.load_data import load_data, node_pairs_among


def hetero2net2(hetero_graph: dgl.DGLHeteroGraph) -> dgl.DGLGraph:
    n_net = hetero_graph.num_nodes(ntype='net')
    us = []
    vs = []
    feats = []

    for nid in hetero_graph.nodes('node'):
        connect_nets = hetero_graph.predecessors(nid, etype='pinned')
        connect_nets = [int(n) for n in connect_nets]
        us_, vs_ = node_pairs_among(connect_nets, max_cap=5)
        feat = hetero_graph.nodes['node'].data['hv'][nid, :]
        us.extend(us_)
        vs.extend(vs_)
        feats.extend([list(feat) for _ in range(len(us_))])

    net_graph = dgl.graph((us, vs), num_nodes=n_net)
    net_graph.ndata['hv'] = hetero_graph.nodes['net'].data['hv']
    net_graph.ndata['label'] = hetero_graph.nodes['net'].data['label']
    net_graph.edata['he'] = torch.tensor(feats, dtype=torch.float32)
    return net_graph


def net2_data(dir_name: str, given_iter, index: int, hashcode: str,
              graph_scale: int = 1000, bin_x: float = 32, bin_y: float = 40, force_save=False, use_tqdm=True,
              app_name='', win_x: float = 32, win_y: float = 40, win_cap=5
              ) -> List[dgl.DGLGraph]:
    file_path = f'{dir_name}/net2{app_name}_{given_iter}.pickle'
    if os.path.exists(file_path) and not force_save:
        with open(file_path, 'rb') as fp:
            list_graph = pickle.load(fp)
        return list_graph

    list_hg_htg = load_data(
        dir_name, given_iter, index, hashcode,
        graph_scale, bin_x, bin_y, False, use_tqdm,
        app_name, win_x, win_y, win_cap
    )
    list_htg = [hg_htg[1] for hg_htg in list_hg_htg]
    if use_tqdm:
        list_graph = [hetero2net2(htg) for htg in tqdm.tqdm(list_htg)]
    else:
        list_graph = [hetero2net2(htg) for htg in list_htg]

    with open(file_path, 'wb+') as fp:
        pickle.dump(list_graph, fp)
    return list_graph

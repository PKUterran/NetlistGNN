import numpy as np
import pickle
import torch
import dgl
import dgl.function as fn
from dgl.transform import add_self_loop, metis_partition

from net.NetlistGNN import NetlistGNN
from data.DIT import dump_data as dump_data_image
from data.DIT import collect_data as collect_data_image
from data.DIT import load_data as load_data_image

DATA_DIR = 'data/test'


def dump_data():
    edge = {
        0: [(1, 0, 0, 0), (2, 0, 0, 0), (0, 0, 50, 1)],
        1: [(0, 0, -50, 0), (3, 0, 0, 1)],
        2: [(5, 0, 0, 0), (4, 0, 0, 1)]
    }
    with open(f'{DATA_DIR}/edge.pkl', 'wb+') as fp:
        pickle.dump(edge, fp, protocol=pickle.HIGHEST_PROTOCOL)
    sizdata_x = np.array([200, 40, 40, 40, 40, 40])
    sizdata_y = np.array([250, 40, 40, 40, 40, 40])
    xdata = np.array([400, 200, 500, 300, 500, 800])
    ydata = np.array([450, 700, 700, 100, 100, 700])
    labels = np.random.random(size=[6, 12])
    np.save(f'{DATA_DIR}/sizdata_x.npy', sizdata_x)
    np.save(f'{DATA_DIR}/sizdata_y.npy', sizdata_y)
    np.save(f'{DATA_DIR}/xdata_900.npy', xdata)
    np.save(f'{DATA_DIR}/ydata_900.npy', ydata)
    np.save(f'{DATA_DIR}/iter_900_node_label_full_000000_.npy', labels)


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


def process_data(dir_name: str, given_iter, index: int, hashcode: str,
                 graph_scale: int = 5, bin_x: float = 500, bin_y: float = 400):
    with open(f'{dir_name}/edge.pkl', 'rb') as fp:
        edge = pickle.load(fp)
    sizdata_x = np.load(f'{dir_name}/sizdata_x.npy')
    sizdata_y = np.load(f'{dir_name}/sizdata_y.npy')
    xdata = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    ydata = np.load(f'{dir_name}/ydata_{given_iter}.npy')
    n_node = len(sizdata_x)
    labels = np.load(f'{dir_name}/iter_{given_iter}_node_label_full_{hashcode}_.npy')
    labels = labels[:n_node, index]

    node_hv = torch.tensor(np.vstack((sizdata_x, sizdata_y)), dtype=torch.float32).t()
    node_pos = torch.tensor(np.vstack((xdata, ydata)), dtype=torch.float32).t()

    # homo_graph
    us, vs = [], []
    for net, list_node_feats in edge.items():
        nodes = [node_feats[0] for node_feats in list_node_feats]
        us_, vs_ = node_pairs_among(nodes)
        us.extend(us_)
        vs.extend(vs_)
    homo_graph = add_self_loop(dgl.graph((us, vs), num_nodes=n_node))
    homo_graph.ndata['pos'] = node_pos[:n_node, :]
    homo_graph.ndata['feat'] = node_hv
    extra = fo_average(homo_graph)
    homo_graph.ndata.pop('inter')
    homo_graph.ndata.pop('addnlfeat')
    homo_graph.ndata.pop('wts')
    homo_graph.ndata.pop('wtmsg')
    homo_graph.ndata['feat'] = torch.cat([homo_graph.ndata['feat'], extra], dim=1)
    homo_graph.ndata['label'] = torch.tensor(labels, dtype=torch.float32)
    # partition_list = get_partition_list_random(homo_graph, int(np.ceil(n_node / graph_scale)))
    partition_list = [[0, 1, 2, 3], [4, 5]]
    list_homo_graph = [dgl.node_subgraph(homo_graph, partition) for partition in partition_list]

    # hetero_graph
    us4, vs4 = [], []
    for x_offset, y_offset in [(0, 0), (bin_x / 2, 0), (0, bin_y / 2), (bin_x / 2, bin_y / 2)]:
        box_node = {}
        iter_sp = enumerate(zip(sizdata_x, sizdata_y, xdata, ydata))
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
        for nodes in box_node.values():
            us_, vs_ = node_pairs_among(nodes, max_cap=5)
            us.extend(us_)
            vs.extend(vs_)
        us4.extend(us)
        vs4.extend(vs)

    print('\thetero_graph generated 1/2')

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
        ('node', 'near', 'node'): (us4, vs4),
        ('node', 'pins', 'net'): (us, vs),
        ('net', 'pinned', 'node'): (vs, us),
    }, num_nodes_dict={'node': n_node, 'net': len(net_degree)})
    hetero_graph.nodes['node'].data['hv'] = homo_graph.ndata['feat']
    hetero_graph.nodes['net'].data['hv'] = net_hv
    # hetero_graph.edges['pins'].data['he'] = torch.tensor(he, dtype=torch.float32)
    hetero_graph.edges['pinned'].data['he'] = torch.tensor(he, dtype=torch.float32)

    list_hetero_graph = []
    for partition in partition_list:
        partition_set = set(partition)
        keep_net_ids = set()
        for net_id, node_id in zip(*[ns.tolist() for ns in hetero_graph.edges(etype='pinned')]):
            if node_id in partition_set:
                keep_net_ids.add(net_id)
        part_hetero_graph = dgl.node_subgraph(hetero_graph, nodes={'node': partition, 'net': list(keep_net_ids)})
        # remove_net_ids = [net_id for net_id in part_hetero_graph.nodes('net')
        #                   if part_hetero_graph.in_degrees(net_id, etype='pins') == 0]
        # part_hetero_graph.remove_nodes(remove_net_ids, ntype='net')
        # print(partition)
        # print(all_net_ids)
        # print(partition)
        # print(keep_net_ids)
        print(part_hetero_graph)
        # print(part_hetero_graph.edges['near'])
        # print(remove_net_ids)
        # exit(123)
        list_hetero_graph.append(part_hetero_graph)

    return list_homo_graph, list_hetero_graph


DEFAULT_CONFIG = {
    'N_LAYER': 2,
    'NODE_FEATS': 16,
    'NET_FEATS': 32,
    'GRID_FEATS': 5,
    'GRID_HEADS': 2,
    'GRID_CHANNELS': 4,
    'PIN_FEATS': 12,
}

if __name__ == '__main__':
    # dump_data()
    _, list_hg = process_data(DATA_DIR, 900, 8, '000000')

    list_hg = [dgl.add_self_loop(hg, etype='near') for hg in list_hg]
    list_hg = [dgl.remove_edges(hg, hg.edges('eid', etype='near'), etype='near') for hg in list_hg]
    # list_hg = [dgl.remove_edges(hg, hg.edges('eid', etype='pinned'), etype='pinned') for hg in list_hg]
    for hg in list_hg:
        print(hg)

    model = NetlistGNN(4, 1, 3, 1, DEFAULT_CONFIG, recurrent=True)
    for i, hg in enumerate(list_hg):
        pred = model.forward(hg.nodes['node'].data['hv'], hg.nodes['net'].data['hv'], hg.edges['pinned'].data['he'],
                             hg)
        print(pred)

    # print(node_pairs_among(list(range(12)), max_cap=10))
    # dump_data_image('data/superblue7_processed', 'data/superblue7', 800, 8, '000000', force_save=False)
    # dump_data_image('data/superblue9_processed', 'data/superblue9', 800, 8, '000000', force_save=False)
    # dump_data_image('data/superblue14_processed', 'data/superblue14', 800, 8, '000000', force_save=False)
    # dump_data_image('data/superblue16_processed', 'data/superblue16', 700, 8, '000000', force_save=False)
    # dump_data_image('data/superblue19_processed', 'data/superblue19', 900, 8, '000000', force_save=False)
    # collect_data_image([
    #     'data/superblue7_processed',
    #     'data/superblue9_processed',
    #     'data/superblue14_processed',
    #     'data/superblue16_processed',
    # ], 'data/train_images', clear_files=True)
    # collect_data_image(['data/superblue19_processed'], 'data/test_images', clear_files=False)
    # _, train_images = load_data_image('data/train_images')
    # _, test_images = load_data_image('data/test_images')
    # print(len(train_images))
    # print(len(test_images))

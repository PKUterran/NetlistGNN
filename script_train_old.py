import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np

from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
import torch.nn as nn

from data.load_data_old import load_data
from net.HyperGNN2D import HyperGNN2D

import warnings

warnings.filterwarnings("ignore")

logs: List[Dict[str, Any]] = []


def printout(arr1, arr2, prefix="", log_prefix=""):
    pearsonr_rho, pearsonr_pval = pearsonr(arr1, arr2)
    spearmanr_rho, spearmanr_pval = spearmanr(arr1, arr2)
    kendalltau_rho, kendalltau_pval = kendalltau(arr1, arr2)
    mae = np.sum(np.abs(arr1 - arr2)) / len(arr1)
    delta = np.abs(arr1 - arr2)
    rmse = np.sqrt(np.sum(np.multiply(delta, delta)) / len(arr1))
    print(prefix + "pearson", pearsonr_rho, pearsonr_pval)
    print(prefix + "spearman", spearmanr_rho, spearmanr_pval)
    print(prefix + "kendall", kendalltau_rho, kendalltau_pval)
    print(prefix + "MAE", mae)
    print(prefix + "RMSE", rmse)
    logs[-1].update({
        f'{log_prefix}pearson_rho': pearsonr_rho,
        f'{log_prefix}pearsonr_pval': pearsonr_pval,
        f'{log_prefix}spearmanr_rho': spearmanr_rho,
        f'{log_prefix}spearmanr_pval': spearmanr_pval,
        f'{log_prefix}kendalltau_rho': kendalltau_rho,
        f'{log_prefix}kendalltau_pval': kendalltau_pval,
        f'{log_prefix}mae': mae,
        f'{log_prefix}rmse': rmse,
    })


def rademacher(intensity, numindices):
    arr = np.random.randint(low=0, high=2, size=numindices)
    return intensity * (2 * arr - 1)


def get_grid_level_corr(posandpred, binx, biny, xgridshape, ygridshape, set_name=''):
    nodetarg, nodepred, posx, posy = [posandpred[:, i] for i in range(0, posandpred.shape[1])]
    cmap_tgt = np.zeros((xgridshape, ygridshape))
    cmap_prd, supp = np.zeros_like(cmap_tgt), np.zeros_like(cmap_tgt)
    wmap = 1e-6 * np.ones_like(cmap_tgt)
    indices = []
    for i in range(0, posandpred.shape[0]):
        key1, key2 = int(np.rint(posx[i] / binx)), int(np.rint(posy[i] / biny))
        wmap[key1][key2] += 1
        indices += [key2 + key1 * ygridshape]
        cmap_prd[key1][key2] += nodepred[i]
        cmap_tgt[key1][key2] += nodetarg[i]
    supp = np.clip(wmap, 0, 1)
    indices = list(set(indices))
    if 0 in indices:
        indices.remove(0)
    cmap_tgt = np.divide(cmap_tgt, wmap)
    cmap_prd = np.divide(cmap_prd, wmap)
    cmap_prd[0, 0] = 0
    cmap_tgt[0, 0] = 0
    wmap[0, 0] = 1e-6
    nctu, pred = cmap_tgt.flatten(), cmap_prd.flatten()
    getmask = np.zeros_like(nctu)
    getmask[indices] = 1
    nctu, pred = np.multiply(nctu, getmask), np.multiply(pred, getmask)
    printout(nctu[indices] + rademacher(1e-6, len(indices)), pred[indices] + rademacher(1e-6, len(indices)),
             "\t\tGRID_INDEX: ", f'{set_name}grid_index_')
    printout(nctu, pred, "\t\tGRID_NO_INDEX: ", f'{set_name}grid_no_index_')
    return


argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='main')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=10)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--weight_decay', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=1e-1)

argparser.add_argument('--layers', type=int, default=2)  # 2
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--grid_feats', type=int, default=16)  # 16
argparser.add_argument('--heads', type=int, default=2)  # 2

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outtype', type=str, default='sig')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)

argparser.add_argument('--graph_scale', type=int, default=10000)
args = argparser.parse_args()

seed = args.seed
np.random.seed(seed)
torch.manual_seed(seed)

device = torch.device(args.device)
if not args.device == 'cpu':
    torch.cuda.set_device(device)
    torch.cuda.manual_seed(seed)

config = {
    'N_LAYER': args.layers,
    'NODE_FEATS': args.node_feats,
    'NET_FEATS': args.net_feats,
    'GRID_FEATS': args.grid_feats,
    'GRID_HEADS': args.heads,
    'GRID_CHANNELS': 4,
    'PIN_FEATS': args.pin_feats,
}

train_dataset_names = [
    'superblue1_processed',
    'superblue2_processed',
    'superblue3_processed',
    'superblue5_processed',
    'superblue6_processed',
    'superblue7_processed',
    'superblue9_processed',
    'superblue11_processed',
    'superblue14_processed',
]
validate_dataset_name = 'superblue16_processed'
test_dataset_name = 'superblue19_processed'

train_list_tuple_graph, test_list_tuple_graph = [], []

for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            train_list_tuple_graph.extend(list_tuple_graph)

for dataset_name in [test_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            test_list_tuple_graph.extend(list_tuple_graph)
n_train_node = sum(map(lambda x: x[0].number_of_nodes(), train_list_tuple_graph))
n_test_node = sum(map(lambda x: x[0].number_of_nodes(), test_list_tuple_graph))

print('##### MODEL #####')
in_node_feats = train_list_tuple_graph[0][1].nodes['node'].data['hv'].shape[1]
if args.grid_feats == 0:
    in_node_feats -= 6
in_net_feats = train_list_tuple_graph[0][1].nodes['net'].data['hv'].shape[1]
in_pin_feats = train_list_tuple_graph[0][1].edges['pinned'].data['he'].shape[1]
model = HyperGNN2D(
    in_node_feats=in_node_feats,
    in_net_feats=in_net_feats,
    in_pin_feats=in_pin_feats,
    n_target=1,
    activation=args.outtype,
    config=config,
).to(device)
n_param = 0
for name, param in model.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)


def to_device(a, b, c):
    return a.to(device), b.to(device), [ci.to(device) for ci in c]


for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
    logs.append({'epoch': epoch})
    t0 = time()
    if epoch:
        model.train()
        for _ in range(args.train_epoch):
            t1 = time()
            losses = []
            n_tuples = len(train_list_tuple_graph)
            for j, (homo_graph, hetero_graph, grid_graphs) in enumerate(train_list_tuple_graph):
                homo_graph, hetero_graph, grid_graphs = to_device(homo_graph, hetero_graph, grid_graphs)
                optimizer.zero_grad()
                pred = model.forward(
                    in_node_feat=hetero_graph.nodes['node'].data['hv'][:, [0, 1, 2, 6, 7, 8]] if args.grid_feats == 0 else hetero_graph.nodes['node'].data['hv'],
                    in_net_feat=hetero_graph.nodes['net'].data['hv'],
                    in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                    node_net_graph=hetero_graph,
                    list_grid_graph=grid_graphs,
                ) * args.scalefac
                batch_labels = homo_graph.ndata['label']
                loss = loss_f(pred.view(-1), batch_labels.float())
                losses.append(loss)
                if len(losses) >= args.batch or j == n_tuples - 1:
                    sum(losses).backward()
                    optimizer.step()
                    losses.clear()
            scheduler.step()
            print(f"\tTraining time per epoch: {time() - t1}")
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    model.eval()


    def evaluate(ltg, set_name, n_node, single_net=False):
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 4))
        p = 0
        with torch.no_grad():
            for z, (hmg, htg, ggs) in enumerate(ltg):
                hmg, htg, ggs = to_device(hmg, htg, ggs)
                # print(hmg.num_nodes(), hmg.num_edges())
                # print([(gg.num_nodes(), gg.num_edges()) for gg in ggs])
                prd = model.forward(
                    in_node_feat=htg.nodes['node'].data['hv'][:, [0, 1, 2, 6, 7, 8]] if args.grid_feats == 0 else htg.nodes['node'].data['hv'],
                    in_net_feat=htg.nodes['net'].data['hv'],
                    in_pin_feat=htg.edges['pinned'].data['he'],
                    node_net_graph=htg,
                    list_grid_graph=ggs,
                ) * args.scalefac
                output_labels = hmg.ndata['label']
                output_pos = (hmg.ndata['pos'].cpu().data.numpy())
                output_predictions = prd
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                ln = len(tgt)
                outputdata[p:p + ln, 0], outputdata[p:p + ln, 1], outputdata[p:p + ln, 2:4] = tgt, prd, output_pos
                p += ln
        outputdata = outputdata[:p, :]
        if single_net:
            get_grid_level_corr(outputdata, args.binx, args.biny,
                                int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                                int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                                set_name=set_name)
        printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_')


    evaluate(train_list_tuple_graph, 'train_', n_train_node)
    evaluate(test_list_tuple_graph, 'test_', n_test_node, single_net=True)
    # exit(123)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

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

from data.load_data import load_data
from net.naive import TraditionalGNNModel

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


argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='GAT')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--lr', type=int, default=1e-3)
argparser.add_argument('--weight_decay', type=int, default=2e-5)
argparser.add_argument('--lr_decay', type=int, default=0.98)

argparser.add_argument('--graph_type', type=str, default='GAT')
argparser.add_argument('--architecture', type=str, default='400,320')
argparser.add_argument('--degdim', type=int, default=0)
argparser.add_argument('--heads', type=str, default='1')

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='000000')
argparser.add_argument('--logic_features', type=bool, default=True)
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--train_epoch', type=int, default=5)
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

if ',' in args.architecture:
    arch = list(map(int, args.architecture.split(',')))
else:
    arch = [(int(args.architecture))]

assert args.graph_type in ['GCN', 'SAGE', 'GAT']

if args.logic_features:
    nfeats = 3
else:
    nfeats = 4

arch.insert(0, 2 * nfeats + args.degdim)
arch.append(1)

train_dataset_names = [
    # 'superblue7_processed',
    'superblue9_processed',
    'superblue14_processed',
    'superblue16_processed',
]
test_dataset_names = [
    'superblue19_processed',
]

train_list_tuple_graph, test_list_tuple_graph = [], []

for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            train_list_tuple_graph.extend(list_tuple_graph)
# exit(123)
for dataset_name in test_dataset_names:
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
in_net_feats = train_list_tuple_graph[0][1].nodes['net'].data['hv'].shape[1]
in_pin_feats = train_list_tuple_graph[0][1].edges['pinned'].data['he'].shape[1]
model = TraditionalGNNModel(
    model_type=args.graph_type,
    arch_detail=arch,
    heads=int(args.heads),
    activation=args.outtype,
    scalefac=args.scalefac,
).to(device)
n_param = 0
for name, param in model.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=args.lr_decay)

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
            for j, (homo_graph, hetero_graph, grid_graphs) in enumerate(train_list_tuple_graph):
                homo_graph, _, _ = to_device(homo_graph, hetero_graph, grid_graphs)
                optimizer.zero_grad()
                pred = model.wholeforward(
                    g=homo_graph,
                    x=homo_graph.ndata['feat'] if args.logic_features
                    else torch.cat([homo_graph.ndata['feat'], homo_graph.ndata['pos']], dim=-1)
                )
                batch_labels = homo_graph.ndata['label']
                loss = loss_f(pred.view(-1), batch_labels.float())
                loss.backward()
                optimizer.step()
            scheduler.step()
            print(f"\tTraining time per epoch: {time() - t1}")
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    model.eval()


    def evaluate(ltg, set_name, n_node):
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 4))
        p = 0
        with torch.no_grad():
            for z, (hmg, htg, ggs) in enumerate(ltg):
                hmg, _, _ = to_device(hmg, htg, ggs)
                prd = model.wholeforward(
                    g=hmg,
                    x=hmg.ndata['feat'] if args.logic_features
                    else torch.cat([hmg.ndata['feat'], hmg.ndata['pos']], dim=-1)
                )
                output_labels = hmg.ndata['label']
                output_pos = (hmg.ndata['pos'].cpu().data.numpy())
                output_predictions = prd
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                ln = len(tgt)
                outputdata[p:p + ln, 0], outputdata[p:p + ln, 1], outputdata[p:p + ln, 2:4] = tgt, prd, output_pos
                p += ln
        outputdata = outputdata[:p, :]
        printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}_node_level_')


    evaluate(train_list_tuple_graph, 'train', n_train_node)
    evaluate(test_list_tuple_graph, 'test', n_test_node)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

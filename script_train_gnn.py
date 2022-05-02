import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np

import torch
import torch.nn as nn

from data.load_data import load_data
from net.naive import TraditionalGNNModel
from utils.output import printout, get_grid_level_corr

import warnings

warnings.filterwarnings("ignore")

logs: List[Dict[str, Any]] = []

argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='GAT')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=25)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--lr', type=int, default=1e-3)
argparser.add_argument('--weight_decay', type=int, default=2e-4)
argparser.add_argument('--lr_decay', type=int, default=3e-2)

argparser.add_argument('--graph_type', type=str, default='GAT')
argparser.add_argument('--architecture', type=str, default='400,320')
argparser.add_argument('--degdim', type=int, default=0)
argparser.add_argument('--heads', type=str, default='1')

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--logic_features', type=bool, default=True)
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

if ',' in args.architecture:
    arch = list(map(int, args.architecture.split(',')))
else:
    arch = [(int(args.architecture))]

assert args.graph_type in ['GCN', 'SAGE', 'GAT']

if args.logic_features:
    nfeats = 3
else:
    nfeats = 8

arch.insert(0, 2 * nfeats + args.degdim)
arch.append(1)

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

train_list_tuple_graph, validate_list_tuple_graph, test_list_tuple_graph = [], [], []

for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            train_list_tuple_graph.extend(list_tuple_graph)
# exit(123)
for dataset_name in [validate_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            validate_list_tuple_graph.extend(list_tuple_graph)
for dataset_name in [test_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False)
            test_list_tuple_graph.extend(list_tuple_graph)
n_train_node = sum(map(lambda x: x[0].number_of_nodes(), train_list_tuple_graph))
n_validate_node = sum(map(lambda x: x[0].number_of_nodes(), validate_list_tuple_graph))
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
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
    logs.append({'epoch': epoch})


    def train(ltg):
        model.train()
        t1 = time()
        for j, (homo_graph, _) in enumerate(ltg):
            homo_graph = homo_graph.to(device)
            optimizer.zero_grad()
            pred = model.wholeforward(
                g=homo_graph,
                x=homo_graph.ndata['feat'][:, [0, 1, 2, 7, 8, 9]] if args.logic_features
                else torch.cat([homo_graph.ndata['feat'], homo_graph.ndata['pos']], dim=-1)
            )
            batch_labels = homo_graph.ndata['label']
            if args.logic_features:
                loss = loss_f(pred.view(-1), batch_labels.float())
            else:
                weight = 1 / homo_graph.ndata['feat'][:, 6]
                weight[torch.isinf(weight)] = 0.
                loss = torch.sum(((pred.view(-1) - batch_labels.float()) ** 2) * weight) / torch.sum(weight)
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")


    def evaluate(ltg, set_name, n_node, single_net=False):
        model.eval()
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 5))
        p = 0
        with torch.no_grad():
            for j, (homo_graph, _) in enumerate(ltg):
                homo_graph = homo_graph.to(device)
                prd = model.wholeforward(
                    g=homo_graph,
                    x=homo_graph.ndata['feat'][:, [0, 1, 2, 7, 8, 9]] if args.logic_features
                    else torch.cat([homo_graph.ndata['feat'], homo_graph.ndata['pos']], dim=-1)
                )
                density = homo_graph.ndata['feat'][:, 6].cpu().data.numpy()
                output_labels = homo_graph.ndata['label']
                output_pos = (homo_graph.ndata['pos'].cpu().data.numpy())
                output_predictions = prd
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                ln = len(tgt)
                outputdata[p:p + ln, 0], outputdata[p:p + ln, 1], outputdata[p:p + ln, 2:4], outputdata[p:p + ln, 4] = \
                    tgt, prd, output_pos, density
                p += ln
        outputdata = outputdata[:p, :]
        if not args.logic_features:
            outputdata[outputdata[:, 4] < 0.5, 1] = outputdata[outputdata[:, 4] < 0.5, 0]
        d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_')
        logs[-1].update(d)
        if single_net:
            d1, d2 = get_grid_level_corr(outputdata[:, :4], args.binx, args.biny,
                                         int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                                         int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                                         set_name=set_name)
            logs[-1].update(d1)
            logs[-1].update(d2)


    t0 = time()
    if epoch:
        model.train()
        for _ in range(args.train_epoch):
            train(train_list_tuple_graph)
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    evaluate(train_list_tuple_graph, 'train_', n_train_node)
    evaluate(validate_list_tuple_graph, 'validate_', n_validate_node, single_net=True)
    evaluate(test_list_tuple_graph, 'test_', n_test_node, single_net=True)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

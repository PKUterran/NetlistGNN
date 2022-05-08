import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np
import dgl

import torch
import torch.nn as nn

from data.net2_data import net2_data
from net.Net2 import MLP, Net2f, Net2a
from log.store_scatter import store_scatter
from utils.output import printout_xf1

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
logs: List[Dict[str, Any]] = []

argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='net2a')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)

argparser.add_argument('--model', type=str, default='net2a')  # True
argparser.add_argument('--hfeats', type=int, default=64)  # 64

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outtype', type=str, default='tanh')
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

train_dataset_names = [
    'superblue_0425_withHPWL/superblue3_processed',
    'superblue_0425_withHPWL/superblue6_processed',
    'superblue_0425_withHPWL/superblue7_processed',
    'superblue_0425_withHPWL/superblue9_processed',
    'superblue_0425_withHPWL/superblue11_processed',
    'superblue_0425_withHPWL/superblue12_processed',
    'superblue_0425_withHPWL/superblue14_processed',
]
validate_dataset_name = 'superblue_0425_withHPWL/superblue16_processed'
test_dataset_name = f'superblue_0425_withHPWL/{args.test}_processed'

train_list_graph, validate_list_graph, test_list_graph = [], [], []

for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = net2_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            train_list_graph.extend(list_tuple_graph)

for dataset_name in [validate_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = net2_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            validate_list_graph.extend(list_tuple_graph)

for dataset_name in [test_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = net2_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            test_list_graph.extend(list_tuple_graph)

print('##### MODEL #####')
nfeats = 1
efeats = train_list_graph[0].edata['he'].shape[1]

if args.model == 'mlp':
    model = MLP(
        nfeats=nfeats,
        hfeats=args.hfeats,
        n_target=1,
        activation=args.outtype,
    ).to(device)
elif args.model == 'net2f':
    model = Net2f(
        nfeats=nfeats,
        hfeats=args.hfeats,
        n_target=1,
        activation=args.outtype,
    ).to(device)
elif args.model == 'net2a':
    model = Net2a(
        nfeats=nfeats,
        efeats=efeats,
        hfeats=args.hfeats,
        n_target=1,
        activation=args.outtype,
    ).to(device)
else:
    assert False, f'Model name: {args.model}'
print(f'Use model: {args.model}')

n_param = 0
for name, param in model.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

if args.beta < 1e-5:
    print(f'### USE L1Loss ###')
    loss_f = nn.L1Loss()
elif args.beta > 7.0:
    print(f'### USE MSELoss ###')
    loss_f = nn.MSELoss()
else:
    print(f'### USE SmoothL1Loss with beta={args.beta} ###')
    loss_f = nn.SmoothL1Loss(beta=args.beta)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=(1 - args.lr_decay))

LOG_DIR = f'log/hpwl-{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
FIG_DIR = 'log/hpwl-temp'
if not os.path.isdir(FIG_DIR):
    os.mkdir(FIG_DIR)


for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
    logs.append({'epoch': epoch})


    def train(ltg):
        model.train()
        t1 = time()
        losses = []
        n_tuples = len(ltg)
        for j, graph in enumerate(ltg):
            graph = graph.to(device)
            graph.ndata['hv'] = graph.ndata['hv'][:, [0]]
            optimizer.zero_grad()
            pred = model.forward(graph)
            pred = pred * args.scalefac
            batch_labels = graph.ndata['label']
            loss = loss_f(pred.view(-1), batch_labels.float())
            losses.append(loss)
            if len(losses) >= args.batch or j == n_tuples - 1:
                sum(losses).backward()
                optimizer.step()
                losses.clear()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")


    def evaluate(ltg, set_name):
        model.eval()
        print(f'\tEvaluate {set_name}:')
        all_tgt = []
        all_prd = []
        with torch.no_grad():
            for j, graph in enumerate(ltg):
                graph = graph.to(device)
                graph.ndata['hv'] = graph.ndata['hv'][:, [0]]
                prd = model.forward(graph)
                prd = prd * args.scalefac
                output_labels = graph.ndata['label']
                output_predictions = prd
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                all_tgt.extend(tgt)
                all_prd.extend(prd)
        all_tgt, all_prd = np.array(all_tgt), np.array(all_prd)
        d = printout_xf1(all_tgt, all_prd, "\t\t", f'{set_name}')
        logs[-1].update(d)
        store_scatter(all_tgt, all_prd, f'{args.name}-{set_name}', epoch=epoch, fig_dir=FIG_DIR)

    t0 = time()
    if epoch:
        for _ in range(args.train_epoch):
            train(train_list_graph)
    logs[-1].update({'train_time': time() - t0})
    t2 = time()
    evaluate(train_list_graph, 'train_')
    evaluate(validate_list_graph, 'validate_')
    evaluate(test_list_graph, 'test_')
    # exit(123)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

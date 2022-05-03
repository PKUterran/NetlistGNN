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

from data.load_data import load_data
from net.NetlistGNN import NetlistGNN
from log.draw_scatter import draw_scatter
from utils.output import printout_xf1

import warnings

warnings.filterwarnings("ignore")
np.set_printoptions(precision=3, suppress=True)
logs: List[Dict[str, Any]] = []

argparser = argparse.ArgumentParser("Training")

argparser.add_argument('--name', type=str, default='main')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=2e-4)
argparser.add_argument('--weight_decay', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=2e-2)
argparser.add_argument('--beta', type=float, default=0.5)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)

argparser.add_argument('--layers', type=int, default=6)  # 2
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--edge_feats', type=int, default=4)  # 4
argparser.add_argument('--topo_geom', type=str, default='both')  # default
argparser.add_argument('--recurrent', type=bool, default=True)  # False
argparser.add_argument('--use_topo_edge', type=bool, default=True)  # True
argparser.add_argument('--use_geom_edge', type=bool, default=True)  # True
argparser.add_argument('--pos_code', type=float, default=0.0)  # 0.0

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

config = {
    'N_LAYER': args.layers,
    'NODE_FEATS': args.node_feats,
    'NET_FEATS': args.net_feats,
    'PIN_FEATS': args.pin_feats,
    'EDGE_FEATS': args.edge_feats,
}

# train_dataset_names = [
#     'superblue_0425_withHPWL/superblue6_processed',
# ]
# validate_dataset_name = 'superblue_0425_withHPWL/superblue6_processed'
# test_dataset_name = f'superblue_0425_withHPWL/superblue6_processed'

train_dataset_names = [
    'superblue_0425_withHPWL/superblue6_processed',
    'superblue_0425_withHPWL/superblue7_processed',
    'superblue_0425_withHPWL/superblue9_processed',
    'superblue_0425_withHPWL/superblue14_processed',
]
validate_dataset_name = 'superblue_0425_withHPWL/superblue16_processed'
test_dataset_name = f'superblue_0425_withHPWL/{args.test}_processed'

train_list_tuple_graph, validate_list_tuple_graph, test_list_tuple_graph = [], [], []


def fit_topo_geom(ltg):
    if args.topo_geom == 'topo':
        ltg = [(g, dgl.remove_edges(hg, hg.edges('eid', etype='near'), etype='near')) for g, hg in ltg]
    elif args.topo_geom == 'geom':
        ltg = [(g, dgl.remove_edges(hg, hg.edges('eid', etype='pinned'), etype='pinned')) for g, hg in ltg]
    ltg = [(g, dgl.add_self_loop(hg, etype='near')) for g, hg in ltg]
    return ltg


for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            list_tuple_graph = fit_topo_geom(list_tuple_graph)
            train_list_tuple_graph.extend(list_tuple_graph)

for dataset_name in [validate_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            list_tuple_graph = fit_topo_geom(list_tuple_graph)
            validate_list_tuple_graph.extend(list_tuple_graph)

for dataset_name in [test_dataset_name]:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                                         graph_scale=args.graph_scale,
                                         bin_x=args.binx, bin_y=args.biny, force_save=False,
                                         app_name=args.app_name,
                                         win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)
            list_tuple_graph = fit_topo_geom(list_tuple_graph)
            test_list_tuple_graph.extend(list_tuple_graph)
n_train_node = sum(map(lambda x: x[0].number_of_nodes(), train_list_tuple_graph))
n_validate_node = sum(map(lambda x: x[0].number_of_nodes(), validate_list_tuple_graph))
n_test_node = sum(map(lambda x: x[0].number_of_nodes(), test_list_tuple_graph))

print('##### MODEL #####')
in_node_feats = train_list_tuple_graph[0][1].nodes['node'].data['hv'].shape[1]
in_net_feats = train_list_tuple_graph[0][1].nodes['net'].data['hv'].shape[1]
in_pin_feats = train_list_tuple_graph[0][1].edges['pinned'].data['he'].shape[1]
if args.topo_geom == 'topo':
    in_node_feats = 6
model = NetlistGNN(
    in_node_feats=in_node_feats,
    in_net_feats=in_net_feats,
    in_pin_feats=in_pin_feats,
    in_edge_feats=1,
    n_target=1,
    activation=args.outtype,
    config=config,
    recurrent=args.recurrent,
    use_topo_edge=args.use_topo_edge, use_geom_edge=args.use_geom_edge
).to(device)
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


def to_device(a, b):
    return a.to(device), b.to(device)


for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f'\tLearning rate: {optimizer.state_dict()["param_groups"][0]["lr"]}')
    logs.append({'epoch': epoch})


    def train(ltg):
        model.train()
        t1 = time()
        losses = []
        n_tuples = len(ltg)
        for j, (homo_graph, hetero_graph) in enumerate(ltg):
            homo_graph, hetero_graph = to_device(homo_graph, hetero_graph)
            optimizer.zero_grad()
            in_node_feat = hetero_graph.nodes['node'].data['hv']
            in_net_feat = hetero_graph.nodes['net'].data['hv']
            if args.pos_code > 1e-5 and args.topo_geom != 'topo':
                in_node_feat += args.pos_code * hetero_graph.nodes['node'].data['pos_code']
            if args.topo_geom == 'topo':
                in_node_feat = in_node_feat[:, [0, 1, 2, 7, 8, 9]]
                in_net_feat = in_net_feat[:, [0]]
            _, pred = model.forward(
                in_node_feat=in_node_feat,
                in_net_feat=in_net_feat,
                in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                in_edge_feat=hetero_graph.edges['near'].data['he'],
                node_net_graph=hetero_graph,
            )
            pred = pred * args.scalefac
            batch_labels = hetero_graph.nodes['net'].data['label']
            degree = hetero_graph.nodes['net'].data['degree'].cpu().data.numpy().flatten()
            new_degree = hetero_graph.nodes['net'].data['new_degree'].cpu().data.numpy().flatten()
            good_net = np.abs(degree - new_degree) < 1e-5
            loss = loss_f(pred.view(-1)[good_net], batch_labels.float()[good_net])
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
            for j, (homo_graph, hetero_graph) in enumerate(ltg):
                homo_graph, hetero_graph = to_device(homo_graph, hetero_graph)
                # print(hmg.num_nodes(), hmg.num_edges())
                in_node_feat = hetero_graph.nodes['node'].data['hv']
                if args.pos_code > 1e-5 and args.topo_geom != 'topo':
                    in_node_feat += args.pos_code * hetero_graph.nodes['node'].data['pos_code']
                if args.topo_geom == 'topo':
                    in_node_feat = in_node_feat[:, [0, 1, 2, 7, 8, 9]]
                _, prd = model.forward(
                    in_node_feat=in_node_feat,
                    in_net_feat=hetero_graph.nodes['net'].data['hv'],
                    in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                    in_edge_feat=hetero_graph.edges['near'].data['he'],
                    node_net_graph=hetero_graph,
                )
                prd = prd * args.scalefac
                output_labels = hetero_graph.nodes['net'].data['label']
                output_predictions = prd
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                degree = hetero_graph.nodes['net'].data['degree'].cpu().data.numpy().flatten()
                new_degree = hetero_graph.nodes['net'].data['new_degree'].cpu().data.numpy().flatten()
                good_net = np.abs(degree - new_degree) < 1e-5
#                 print(np.sum(good_net), len(degree))
#                 exit(123)
                all_tgt.extend(tgt[good_net])
                all_prd.extend(prd[good_net])
        all_tgt, all_prd = np.array(all_tgt), np.array(all_prd)
        d = printout_xf1(all_tgt, all_prd, "\t\t", f'{set_name}')
        logs[-1].update(d)
        draw_scatter(all_tgt, all_prd, f'{args.name}-{set_name}', fig_dir=FIG_DIR)

    t0 = time()
    if epoch:
        for _ in range(args.train_epoch):
            train(train_list_tuple_graph)
    logs[-1].update({'train_time': time() - t0})
    t2 = time()
    evaluate(train_list_tuple_graph, 'train_')
    evaluate(validate_list_tuple_graph, 'validate_')
    evaluate(test_list_tuple_graph, 'test_')
    # exit(123)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

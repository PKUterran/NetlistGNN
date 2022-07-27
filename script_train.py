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
from log.store_cong import store_cong_from_node
from utils.output import printout, get_grid_level_corr

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

argparser.add_argument('--layers', type=int, default=2)  # 2
argparser.add_argument('--node_feats', type=int, default=64)  # 64
argparser.add_argument('--net_feats', type=int, default=128)  # 128
argparser.add_argument('--pin_feats', type=int, default=16)  # 16
argparser.add_argument('--edge_feats', type=int, default=4)  # 4
argparser.add_argument('--topo_geom', type=str, default='both')  # default
argparser.add_argument('--add_pos', type=bool, default=False)  # False
argparser.add_argument('--recurrent', type=bool, default=False)  # False
argparser.add_argument('--topo_conv_type', type=str, default='CFCNN')  # CFCNN
argparser.add_argument('--geom_conv_type', type=str, default='SAGE')  # SAGE
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
test_dataset_name = f'{args.test}_processed'

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
    in_net_feats = 1
if args.add_pos:
    in_node_feats += 2
model = NetlistGNN(
    in_node_feats=in_node_feats,
    in_net_feats=in_net_feats,
    in_pin_feats=in_pin_feats,
    in_edge_feats=1,
    n_target=1,
    activation=args.outtype,
    config=config,
    recurrent=args.recurrent,
    topo_conv_type=args.topo_conv_type, geom_conv_type=args.geom_conv_type
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

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
FIG_DIR = 'log/temp'
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
            if args.add_pos:
                in_node_feat = torch.cat([in_node_feat, homo_graph.ndata['pos']], dim=-1)
            pred, _ = model.forward(
                in_node_feat=in_node_feat,
                in_net_feat=in_net_feat,
                in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                in_edge_feat=hetero_graph.edges['near'].data['he'],
                node_net_graph=hetero_graph,
            )
            pred = pred * args.scalefac
            batch_labels = homo_graph.ndata['label']
            weight = 1 / hetero_graph.nodes['node'].data['hv'][:, 6]
            weight[torch.isinf(weight)] = 0.0
            #             weight[weight < 0.201] = 0.0
            #             weight[weight > 0.499] = 0.0
            #             print(weight)
            #             print(homo_graph.ndata['pos'])
            #             exit(123)
            if args.topo_geom != 'topo':
                #                 loss = loss_f(pred.view(-1), batch_labels.float())
                #                 loss = torch.sum(torch.relu((pred.view(-1) - batch_labels.float()) ** 2 - 0.01) * weight) / torch.sum(weight)
                loss = torch.sum(((pred.view(-1) - batch_labels.float()) ** 2) * weight) / torch.sum(weight)
            #                 loss = torch.sum(((pred.view(-1) - batch_labels.float()) ** 2) * (weight ** 2)) / torch.sum((weight ** 2))
            #                 loss = torch.sum(torch.abs(pred.view(-1) - batch_labels.float()) * (weight ** 3)) / torch.sum((weight ** 3))
            else:
                loss = loss_f(pred.view(-1), batch_labels.float())
            losses.append(loss)
            if len(losses) >= args.batch or j == n_tuples - 1:
                sum(losses).backward()
                optimizer.step()
                losses.clear()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")


    def evaluate(ltg, set_name, n_node, single_net=False):
        model.eval()
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 5))
        p = 0
        with torch.no_grad():
            for j, (homo_graph, hetero_graph) in enumerate(ltg):
                homo_graph, hetero_graph = to_device(homo_graph, hetero_graph)
                # print(hmg.num_nodes(), hmg.num_edges())
                in_node_feat = hetero_graph.nodes['node'].data['hv']
                in_net_feat = hetero_graph.nodes['net'].data['hv']
                if args.pos_code > 1e-5 and args.topo_geom != 'topo':
                    in_node_feat += args.pos_code * hetero_graph.nodes['node'].data['pos_code']
                if args.topo_geom == 'topo':
                    in_node_feat = in_node_feat[:, [0, 1, 2, 7, 8, 9]]
                    in_net_feat = in_net_feat[:, [0]]
                if args.add_pos:
                    in_node_feat = torch.cat([in_node_feat, homo_graph.ndata['pos']], dim=-1)
                prd, _ = model.forward(
                    in_node_feat=in_node_feat,
                    in_net_feat=in_net_feat,
                    in_pin_feat=hetero_graph.edges['pinned'].data['he'],
                    in_edge_feat=hetero_graph.edges['near'].data['he'],
                    node_net_graph=hetero_graph,
                )
                prd = prd * args.scalefac
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
        #         print(f'\t\ttarget/predict: {np.max(outputdata[:, 0]):.3f}, {np.max(outputdata[:, 1]):.3f}')
        #         exit(123)
        # if args.topo_geom != 'topo':
        #     bad_node = np.logical_or(outputdata[:, 4] < 0.5, outputdata[:, 4] > 17.5)
        #                 bad_node = outputdata[:, 4] < 2.5
        #     outputdata[bad_node, 1] = outputdata[bad_node, 0]
        #             outputdata = outputdata[np.logical_and(outputdata[:, 4] > 0, outputdata[:, 4] < 5), :]
        #             outputdata = outputdata[outputdata[:, 4] > 5, :]
        #         worst = np.argpartition(np.abs(outputdata[:, 0] - outputdata[:, 1]),-5)[-5:]
        #         print(f'\t\tworst:\n{outputdata[worst, :]}')
        d = printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}node_level_')
        logs[-1].update(d)
        if single_net:
            if set_name == 'test_' and args.test == 'superblue19':
                store_cong_from_node(outputdata[:, 0], outputdata[:, 1], outputdata[:, 2], outputdata[:, 3],
                                     args.binx, args.biny, [321, 518],
                                     f'{args.name}-{set_name}', epoch=epoch, fig_dir=FIG_DIR)
            d1, d2 = get_grid_level_corr(outputdata[:, :4], args.binx, args.biny,
                                         int(np.rint(np.max(outputdata[:, 2]) / args.binx)) + 1,
                                         int(np.rint(np.max(outputdata[:, 3]) / args.biny)) + 1,
                                         set_name=set_name)
            logs[-1].update(d1)
            logs[-1].update(d2)


    t0 = time()
    if epoch:
        for _ in range(args.train_epoch):
            train(train_list_tuple_graph)
    logs[-1].update({'train_time': time() - t0})
    t2 = time()
    evaluate(train_list_tuple_graph, 'train_', n_train_node)
    evaluate(validate_list_tuple_graph, 'validate_', n_validate_node, single_net=True)
    evaluate(test_list_tuple_graph, 'test_', n_test_node, single_net=True)
    # exit(123)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

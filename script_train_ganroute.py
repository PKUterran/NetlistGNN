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

from data.DIT import load_data
from net.GanRoute import ImageAutoEncoder, Discriminator

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
argparser.add_argument('--name', type=str, default='hyper')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--batch', type=int, default=1)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--weight_decay', type=float, default=1e-5)
argparser.add_argument('--weight_decay_dis', type=float, default=5e-4)
argparser.add_argument('--lr_decay', type=float, default=0.98)

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='000000')
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

train_input_images, train_output_images = load_data('data/train_images')
test_input_images, test_output_images = load_data('data/test_images')

print('##### MODEL #####')
print('Generator:')
generator = ImageAutoEncoder().to(device)
n_param = 0
for name, param in generator.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

print('Discriminator:')
discriminator = Discriminator().to(device)
n_param = 0
for name, param in discriminator.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

loss_f = nn.MSELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_dis = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay_dis)

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
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
                    in_node_feat=hetero_graph.nodes['node'].data['hv'],
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
            print(f"\tTraining time per epoch: {time() - t1}")
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    model.eval()

    def train_gen():
        generator.train()
        discriminator.eval()
        optimizer_gen.zero_grad()

    def train_dis():
        generator.eval()
        discriminator.train()
        optimizer_dis.zero_grad()

    def evaluate(ltg, set_name, n_node):
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 4))
        p = 0
        with torch.no_grad():
            for z, (hmg, htg, ggs) in enumerate(ltg):
                hmg, htg, ggs = to_device(hmg, htg, ggs)
                prd = model.forward(
                    in_node_feat=htg.nodes['node'].data['hv'],
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
        printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}_node_level_')


    evaluate(train_list_tuple_graph, 'train', n_train_node)
    evaluate(test_list_tuple_graph, 'test', n_test_node)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

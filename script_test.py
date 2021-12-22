import json
import os
import random
import string
import argparse
from time import time
from typing import List, Dict, Any

import numpy as np

import dgl

from scipy.stats import pearsonr, spearmanr, kendalltau

import torch
import torch.nn as nn

from data.load_data import prepare_data
from net.naive import TraditionalGNNModel
from utils.sampler import ClusterIter

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


def get_random_string(length):
    letters = string.ascii_lowercase
    result_str = ''.join(random.choice(letters) for i in range(length))
    return result_str


def rademacher(intensity, numindices):
    arr = np.random.randint(low=0, high=2, size=numindices)
    return intensity * (2 * arr - 1)


def get_grid_level_corr(posandpred, binx, biny, xgridshape, ygridshape):
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
             "\t\tGRID_INDEX: ", 'grid_index_')
    printout(nctu, pred, "\t\tGRID_NO_INDEX: ", 'grid_no_index_')
    return


argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='default')
argparser.add_argument('--basedatadir', type=str, default='data/')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--architecture', type=str, default='200,160')
argparser.add_argument('--degdim', type=int, default=0)
argparser.add_argument('--batch_size_GCN', type=int, default=256)
argparser.add_argument('--clusters_per_graph', type=int, default=100)
argparser.add_argument('--clusters_per_batch', type=int, default=1)
argparser.add_argument('--graph_type', type=str, default='SAGE')
argparser.add_argument('--heads', type=str, default='1')
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='000000')
argparser.add_argument('--logic_features', type=bool, default=True)
argparser.add_argument('--normalized_labels', type=bool, default=False)
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--train_epoch', type=int, default=5)
argparser.add_argument('--mintrainidx', type=int, default=19)
argparser.add_argument('--maxtrainidx', type=int, default=20)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--scalefac', type=float, default=7.0)
argparser.add_argument('--outscalefac', type=float, default=7.5)
argparser.add_argument('--edgecap', type=int, default=10)
argparser.add_argument('--outtype', type=str, default='sig')
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)
argparser.add_argument('--xshape', type=int, default=321)
argparser.add_argument('--yshape', type=int, default=518)

args = argparser.parse_args()
device = torch.device(args.device)
if not args.device == 'cpu':
    torch.cuda.set_device(device)

if ',' in args.architecture:
    arch = list(map(int, args.architecture.split(',')))
else:
    arch = [(int(args.architecture))]

assert args.graph_type in ['GCN', 'SAGE', 'GAT']

if args.logic_features:
    nfeats = 3
else:
    nfeats = 5

arch.insert(0, 2 * nfeats + args.degdim)
arch.append(1)

bg = []
found_graphs = 0

for suff in range(args.mintrainidx, args.maxtrainidx):
    for i in range(0, args.itermax):
        if (os.path.isfile(args.basedatadir + 'superblue' + str(suff) + '_processed/iter_' + str(
                i) + '_node_label_full_' + args.hashcode + '_.npy')):
            g = prepare_data(args.basedatadir + 'superblue' + str(suff) + '_processed/', i, args.idx,
                             args.normalized_labels, args.outscalefac, args.logic_features, args.hashcode, args.edgecap,
                             args.degdim)
            found_graphs += 1
            bg += [g]

for i in range(args.mintrainidx, args.itermax):
    if (os.path.isfile(args.basedatadir + args.test + '_processed/iter_' + str(
            i) + '_node_label_full_' + args.hashcode + '_.npy')):
        test_graph = prepare_data(args.basedatadir + args.test + '_processed/', i, args.idx, args.normalized_labels,
                                  args.outscalefac, args.logic_features, args.hashcode, args.edgecap, args.degdim)

# rstr1, rstr2 = get_random_string(10), get_random_string(10)
rstr1, rstr2 = 'fycgckbesg', 'wdrasukoue'

bg = dgl.batch(bg)

cluster_iterator = ClusterIter("circuit" + rstr1, bg, args.clusters_per_graph * found_graphs, args.clusters_per_batch,
                               range(bg.number_of_nodes()), use_pp=0)
test_cluster_iterator = ClusterIter("circuit_new" + rstr2, test_graph, args.clusters_per_graph, args.clusters_per_batch,
                                    range(test_graph.number_of_nodes()), use_pp=0)

model = TraditionalGNNModel(args.graph_type, arch, int(args.heads), args.outtype, args.scalefac).to(device)
loss_f = nn.MSELoss()

opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(opt, 1, gamma=0.995)

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f'\tLearning rate: {opt.state_dict()["param_groups"][0]["lr"]}')
    logs.append({'epoch': epoch})
    t0 = time()
    if epoch:
        model.train()
        for _ in range(args.train_epoch):
            t1 = time()
            for j, cluster in enumerate(cluster_iterator):
                cluster = cluster.to(device)
                opt.zero_grad()
                pred = model.wholeforward(cluster, cluster.ndata['feat'])
                batch_labels = cluster.ndata['label']
                loss = loss_f(pred, batch_labels.float())
                loss.backward()
                opt.step()
            scheduler.step()
            print(f"\tTraining time per epoch: {time() - t1}")
    logs[-1].update({'train_time': time() - t0})
    outputdata = np.zeros((test_graph.number_of_nodes(), 4))
    i = 0
    t2 = time()
    model.eval()
    with torch.no_grad():
        for z, cluster_z in enumerate(test_cluster_iterator):
            cluster_z = cluster_z.to(device)
            pred = model.wholeforward(cluster_z, cluster_z.ndata['feat'])
            output_labels = cluster_z.ndata['label']
            output_pos = (cluster_z.ndata['rawpos'].cpu().data.numpy())
            output_predictions = pred
            tgt = output_labels.cpu().data.numpy().flatten()
            prd = output_predictions.cpu().data.numpy().flatten()
            l = len(tgt)
            outputdata[i:i + l, 0], outputdata[i:i + l, 1], outputdata[i:i + l, 2:4] = tgt, prd, output_pos
            i += l
    outputdata = outputdata[:i, :]
    get_grid_level_corr(outputdata, args.binx, args.biny, args.xshape, args.yshape)
    printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", 'node_level_')
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)


import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np
import tqdm

import torch
import torch.nn as nn

from data.LHNN_data import load_data, SparseBinaryMatrix
from net.LHNN import LHNN
from utils.output import printout

import warnings

# torch.autograd.set_detect_anomaly(True)
warnings.filterwarnings("ignore")

logs: List[Dict[str, Any]] = []

argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='LHNN')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=20)
argparser.add_argument('--train_batch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=128)
argparser.add_argument('--lr', type=float, default=1e-3)
argparser.add_argument('--lr_decay', type=float, default=1e-1)
argparser.add_argument('--dim', type=float, default=32)

argparser.add_argument('--seed', type=int, default=0)
argparser.add_argument('--device', type=str, default='cuda:0')
argparser.add_argument('--hashcode', type=str, default='100000')
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
validate_dataset_names = [
    'superblue16_processed',
]
test_dataset_names = [
    f'{args.test}_processed',
]

train_list_tensors, validate_list_tensors, test_list_tensors = [], [], []

for dataset_name in train_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i)
            train_list_tensors.extend(list_tuple_graph)
# exit(123)
for dataset_name in validate_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i)
            validate_list_tensors.extend(list_tuple_graph)
for dataset_name in test_dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            list_tuple_graph = load_data(f'data/{dataset_name}', i)
            test_list_tensors.extend(list_tuple_graph)
n_train_node = sum(map(lambda x: x[1].shape[0], train_list_tensors))
n_validate_node = sum(map(lambda x: x[1].shape[0], validate_list_tensors))
n_test_node = sum(map(lambda x: x[1].shape[0], test_list_tensors))

print('##### MODEL #####')
model = LHNN(4, 4, dim=args.dim).to(device)
n_param = 0
for name, param in model.named_parameters():
    print(f'\t{name}: {param.shape}')
    n_param += reduce(lambda x, y: x * y, param.shape)
print(f'# of parameters: {n_param}')

loss_f = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=(1 - args.lr_decay))

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
        for j, (v_n, v_c, a_cc_, h_nc_, labels, mask) in enumerate(ltg):
            a_cc, h_nc = a_cc_.dense(), h_nc_.dense()
            optimizer.zero_grad()
            g_nc, g_cn, na_cc = model.generate_adj(a_cc, h_nc)
            v_n, v_c, g_nc, g_cn, na_cc, labels = \
                v_n.to(device), v_c.to(device), g_nc.to(device), g_cn.to(device), na_cc.to(device), labels.to(device)
            pred, _ = model.forward(v_n, v_c, g_nc, g_cn, na_cc)
            pred = pred * args.scalefac
            loss = loss_f(pred.view(-1), labels.float())
            loss.backward()
            optimizer.step()
        scheduler.step()
        print(f"\tTraining time per epoch: {time() - t1}")


    def evaluate(ltg, set_name, n_node, single_net=False):
        print(f'\tEvaluate {set_name}:')
        model.eval()
        output_data = np.zeros((n_node, 3))
        p = 0
        with torch.no_grad():
            for j, (v_n, v_c, a_cc_, h_nc_, labels, mask) in enumerate(ltg):
                a_cc, h_nc = a_cc_.dense(), h_nc_.dense()
                g_nc, g_cn, na_cc = model.generate_adj(a_cc, h_nc)
                v_n, v_c, g_nc, g_cn, na_cc = \
                    v_n.to(device), v_c.to(device), g_nc.to(device), g_cn.to(device), na_cc.to(device)
                pred, _ = model.forward(v_n, v_c, g_nc, g_cn, na_cc)
                pred = pred * args.scalefac
                tgt = labels.cpu().data.numpy().flatten()
                prd = pred.cpu().data.numpy().flatten()
                msk = mask.cpu().data.numpy().flatten()
                ln = len(tgt)
                output_data[p:p + ln, 0], output_data[p:p + ln, 1], output_data[p:p + ln, 2] = tgt, prd, msk
                p += ln
        output_data = output_data[:p, :]
        d = printout(output_data[:, 0], output_data[:, 1], "\t\tGRID_NO_INDEX: ", f'{set_name}grid_no_index_')
        logs[-1].update(d)
        d = printout(output_data[output_data[:, 2] > 0, 0], output_data[output_data[:, 2] > 0, 1],
                     "\t\tGRID_INDEX: ", f'{set_name}grid_index_')
        logs[-1].update(d)


    t0 = time()
    if epoch:
        for _ in range(args.train_epoch):
            train(train_list_tensors)
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    evaluate(train_list_tensors, 'train_', n_train_node)
    evaluate(validate_list_tensors, 'validate_', n_validate_node, single_net=True)
    evaluate(test_list_tensors, 'test_', n_test_node, single_net=True)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

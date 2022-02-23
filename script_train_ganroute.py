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
from torch.utils.data import DataLoader

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

_, train_output_images = load_data('data/train_images')
_, test_output_images = load_data('data/test_images')
train_loader = DataLoader(train_output_images, batch_size=8, shuffle=True)
test_loader = DataLoader(test_output_images, batch_size=8)

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

loss_f = nn.BCELoss()
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay)
optimizer_dis = torch.optim.Adam(generator.parameters(), lr=args.lr, weight_decay=args.weight_decay_dis)
real_label = 1.
fake_label = 0.

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)

for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    logs.append({'epoch': epoch})

    def train(data_loader: DataLoader):
        for i, (batch_output_image, _) in enumerate(data_loader):
            discriminator.zero_grad()
            batch_output_image = batch_output_image.to(device)
            batch_input_image = batch_output_image.clone()
            batch_input_image[:, 0, :, :] = 0

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            b_size = batch_output_image.shape[0]
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            pred = discriminator(batch_output_image).view(-1)
            err_dis_real = loss_f(pred, label)
            err_dis_real.backward()
            dis_x = pred.mean().item()

            # Train with all-fake batch
            batch_pred_image = generator(batch_input_image)
            label.fill_(fake_label)
            pred = discriminator(batch_pred_image.detach()).view(-1)
            err_dis_fake = loss_f(pred, label)
            err_dis_fake.backward()
            dis_gen_z1 = pred.mean().item()
            err_dis = err_dis_real + err_dis_fake
            optimizer_dis.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(batch_pred_image).view(-1)
            err_gen = loss_f(output, label)
            err_gen.backward()
            dis_gen_z2 = output.mean().item()
            optimizer_gen.step()

            if i % 12 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, i, len(data_loader),
                         err_dis.item(), err_gen.item(), dis_x, dis_gen_z1, dis_gen_z2))

    def evaluate(data_loader: DataLoader, set_name, n_node):
        print(f'\tEvaluate {set_name}:')
        outputdata = np.zeros((n_node, 4))
        p = 0
        with torch.no_grad():
            for i, (batch_output_image, _) in enumerate(data_loader):
                batch_output_image = batch_output_image.to(device)
                batch_input_image = batch_output_image.clone()
                batch_input_image[:, 0, :, :] = 0
                batch_pred_image = generator(batch_input_image)
                output_labels = batch_output_image[:, 0, :, :]
                output_pos = (hmg.ndata['pos'].cpu().data.numpy())
                output_predictions = batch_pred_image[:, 0, :, :]
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                ln = len(tgt)
                outputdata[p:p + ln, 0], outputdata[p:p + ln, 1], outputdata[p:p + ln, 2:4] = tgt, prd, output_pos
                p += ln
        outputdata = outputdata[:p, :]
        printout(outputdata[:, 0], outputdata[:, 1], "\t\tNODE_LEVEL: ", f'{set_name}_node_level_')


    t0 = time()
    if epoch:
        train(train_loader)
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    evaluate(train_loader, 'train', n_train_node)
    evaluate(test_loader, 'test', n_test_node)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

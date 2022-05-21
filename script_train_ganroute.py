import json
import os
import argparse
from time import time
from typing import List, Dict, Any
from functools import reduce

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from data.DIT import load_data
from net.GanRoute import ImageAutoEncoder, Discriminator
from utils.output import printout
from log.store_cong import store_cong_from_grid_ganroute

import warnings

warnings.filterwarnings("ignore")

logs: List[Dict[str, Any]] = []

argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--name', type=str, default='GanRoute')
argparser.add_argument('--test', type=str, default='superblue19')
argparser.add_argument('--epochs', type=int, default=100)
argparser.add_argument('--train_batch', type=int, default=5)
argparser.add_argument('--batch', type=int, default=128)
argparser.add_argument('--lr', type=float, default=2e-4)
argparser.add_argument('--lr_decay', type=float, default=1e-2)
argparser.add_argument('--gan_lambda', type=float, default=0)
argparser.add_argument('--l1_lambda', type=float, default=10)

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

_, train_output_images = load_data('data/train_images')
_, test_output_images = load_data('data/test_images')
n_train_sample = len(train_output_images)
n_test_sample = len(test_output_images)
train_loader = DataLoader(train_output_images, batch_size=args.batch, shuffle=True)
test_loader = DataLoader(test_output_images, batch_size=args.batch)

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
optimizer_gen = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)
optimizer_dis = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-4)
lr_scheduler_gen = torch.optim.lr_scheduler.StepLR(optimizer_gen, step_size=1, gamma=(1 - args.lr_decay))
real_label = 1.
fake_label = 0.

LOG_DIR = f'log/{args.test}'
if not os.path.isdir(LOG_DIR):
    os.mkdir(LOG_DIR)
FIG_DIR = 'log/temp'
if not os.path.isdir(FIG_DIR):
    os.mkdir(FIG_DIR)

for epoch in range(0, args.epochs + 1):
    print(f'##### EPOCH {epoch} #####')
    print(f"\tLearning rate of generator: {optimizer_gen.state_dict()['param_groups'][0]['lr']}")
    logs.append({'epoch': epoch})


    def train(data_loader: DataLoader):
        for i, (batch_output_image, _) in enumerate(data_loader):
            batch_output_image = batch_output_image.to(device)
            batch_input_image = batch_output_image.clone()
            batch_input_image[:, 0, :, :] = 0

            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # Train with all-real batch
            # discriminator.train()
            # generator.eval()
            discriminator.zero_grad()
            # optimizer_dis.zero_grad()
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
            # discriminator.eval()
            # generator.train()
            generator.zero_grad()
            # optimizer_gen.zero_grad()
            label.fill_(real_label)
            output = discriminator(batch_pred_image).view(-1)
            err_gen_1 = loss_f(output, label)
            err_gen_2 = F.l1_loss(batch_pred_image, batch_output_image)
            err_gen = err_gen_1 * args.gan_lambda + err_gen_2 * args.l1_lambda
            err_gen.backward()
            dis_gen_z2 = output.mean().item()
            optimizer_gen.step()

            if i % 1 == 0:
                print('\t[%d/%d][%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f / %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                      % (epoch, args.epochs, epoch_i, args.train_batch, i, len(data_loader),
                         err_dis.item(), err_gen_1.item(), err_gen_2.item(), dis_x, dis_gen_z1, dis_gen_z2))


    def evaluate(data_loader: DataLoader, set_name, n_sample, single_net=False):
        print(f'\tEvaluate {set_name}:')
        output_data = np.zeros((n_sample * 32 * 32, 3))
        p = 0
        with torch.no_grad():
            for i, (batch_output_image, _) in enumerate(data_loader):
                batch_output_image = batch_output_image.to(device)
                batch_input_image = batch_output_image.clone()
                batch_input_image[:, 0, :, :] = 0
                batch_pred_image = generator(batch_input_image)
                output_labels = batch_output_image[:, 0, ::2, ::2]
                output_predictions = batch_pred_image[:, 0, ::2, ::2]
                output_grid_mask = batch_output_image[:, 0, ::2, 1::2]
                tgt = output_labels.cpu().data.numpy().flatten()
                prd = output_predictions.cpu().data.numpy().flatten()
                mask = output_grid_mask.cpu().data.numpy().flatten()
                ln = len(tgt)
                output_data[p:p + ln, 0], output_data[p:p + ln, 1], output_data[p:p + ln, 2] = tgt, prd, mask
                p += ln
        output_data = output_data[:p, :]
        printout(output_data[:, 0], output_data[:, 1], "\t\tGRID_NO_INDEX: ", f'{set_name}grid_no_index_')
        printout(output_data[output_data[:, 2] > 0, 0], output_data[output_data[:, 2] > 0, 1],
                 "\t\tGRID_INDEX: ", f'{set_name}grid_index_')
        if set_name == 'test_' and args.test == 'superblue19':
            store_cong_from_grid_ganroute(output_data[:, 0], output_data[:, 1], 32, 32, [160, 288],
                                          f'{args.name}-{set_name}', epoch=epoch, fig_dir=FIG_DIR)

    t0 = time()
    if epoch:
        for epoch_i in range(args.train_batch):
            train(train_loader)
        lr_scheduler_gen.step()
    logs[-1].update({'train_time': time() - t0})

    t2 = time()
    evaluate(train_loader, 'train_', n_train_sample)
    evaluate(test_loader, 'test_', n_test_sample, single_net=True)
    print("\tinference time", time() - t2)
    logs[-1].update({'eval_time': time() - t2})
    with open(f'{LOG_DIR}/{args.name}.json', 'w+') as fp:
        json.dump(logs, fp)

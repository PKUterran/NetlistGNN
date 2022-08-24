import os
import argparse

from data.load_data import load_data

import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--hashcode', type=str, default='100000')
argparser.add_argument('--graph_scale', type=int, default=10000)
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)

argparser.add_argument('--app_name', type=str, default='')
argparser.add_argument('--win_x', type=float, default=32)
argparser.add_argument('--win_y', type=float, default=40)
argparser.add_argument('--win_cap', type=int, default=5)
args = argparser.parse_args()

dataset_names = [
    'superblue1',
    'superblue2',
    'superblue3',
    'superblue5',
    'superblue6',
    'superblue7',
    'superblue9',
    'superblue11',
    'superblue14',
    'superblue16',
    'superblue19',
]

for dataset_name in dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                      graph_scale=args.graph_scale,
                      bin_x=args.binx, bin_y=args.biny, force_save=True, use_tqdm=True,
                      app_name=args.app_name,
                      win_x=args.win_x, win_y=args.win_y, win_cap=args.win_cap)

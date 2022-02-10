import os
import argparse

from data.load_data import load_data

import warnings

warnings.filterwarnings("ignore")

argparser = argparse.ArgumentParser("Training")
argparser.add_argument('--idx', type=int, default=8)
argparser.add_argument('--itermax', type=int, default=2500)
argparser.add_argument('--hashcode', type=str, default='000000')
argparser.add_argument('--graph_scale', type=int, default=3000)
argparser.add_argument('--binx', type=int, default=32)
argparser.add_argument('--biny', type=int, default=40)
args = argparser.parse_args()

dataset_names = [
    'superblue7_processed',
    'superblue9_processed',
    'superblue14_processed',
    'superblue16_processed',
    'superblue19_processed',
]

for dataset_name in dataset_names:
    for i in range(0, args.itermax):
        if os.path.isfile(f'data/{dataset_name}/iter_{i}_node_label_full_{args.hashcode}_.npy'):
            print(f'Loading {dataset_name}:')
            load_data(f'data/{dataset_name}', i, args.idx, args.hashcode,
                      graph_scale=args.graph_scale,
                      bin_x=args.binx, bin_y=args.biny, force_save=False, use_tqdm=True)

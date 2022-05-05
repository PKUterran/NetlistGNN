import os
import argparse

from data.load_data import load_data
from data.LHNN_data import dump_data as dump_data_lhnn
from data.LHNN_data import load_data as load_data_lhnn

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
#     'superblue_0425_withHPWL/superblue6_processed',
#     'superblue_0425_withHPWL/superblue7_processed',
#     'superblue_0425_withHPWL/superblue9_processed',
#     'superblue_0425_withHPWL/superblue14_processed',
#     'superblue_0425_withHPWL/superblue16_processed',
#     'superblue_0425_withHPWL/superblue19_processed',
    
#     'superblue_0425_withHPWL/superblue2_processed',
    'superblue_0425_withHPWL/superblue3_processed',
    'superblue_0425_withHPWL/superblue11_processed',
    'superblue_0425_withHPWL/superblue12_processed',
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

# dump_data_lhnn('data/superblue_0425_withHPWL/superblue6_processed',
#                'data/superblue_0425_withHPWL/superblue6', 811, 8, '100000', force_save=True, enable_net_label=True)
# dump_data_lhnn('data/superblue_0425_withHPWL/superblue7_processed',
#                'data/superblue_0425_withHPWL/superblue7', 800, 8, '100000', force_save=True, enable_net_label=True)
# dump_data_lhnn('data/superblue_0425_withHPWL/superblue9_processed',
#                'data/superblue_0425_withHPWL/superblue9', 800, 8, '100000', force_save=True, enable_net_label=True)
# dump_data_lhnn('data/superblue_0425_withHPWL/superblue14_processed',
#                'data/superblue_0425_withHPWL/superblue14', 700, 8, '100000', force_save=True, enable_net_label=True)
# dump_data_lhnn('data/superblue_0425_withHPWL/superblue16_processed',
#                'data/superblue_0425_withHPWL/superblue16', 700, 8, '100000', force_save=True, enable_net_label=True)
# dump_data_lhnn('data/superblue_0425_withHPWL/superblue19_processed',
#                'data/superblue_0425_withHPWL/superblue19', 700, 8, '100000', force_save=True, enable_net_label=True)

# dump_data_lhnn('data/superblue_0425_withHPWL/superblue2_processed',
#                'data/superblue_0425_withHPWL/superblue2', 586, 8, '100000', force_save=True, enable_net_label=True)
dump_data_lhnn('data/superblue_0425_withHPWL/superblue3_processed',
               'data/superblue_0425_withHPWL/superblue3', 800, 8, '100000', force_save=True, enable_net_label=True)
dump_data_lhnn('data/superblue_0425_withHPWL/superblue11_processed',
               'data/superblue_0425_withHPWL/superblue11', 900, 8, '100000', force_save=True, enable_net_label=True)
dump_data_lhnn('data/superblue_0425_withHPWL/superblue12_processed',
               'data/superblue_0425_withHPWL/superblue12', 1300, 8, '100000', force_save=True, enable_net_label=True)

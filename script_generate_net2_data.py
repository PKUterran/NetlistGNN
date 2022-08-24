import os
import argparse

from data.net2_data import net2_data

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    net2_data('data/superblue_0425_withHPWL/superblue6', 811, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue7', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue9', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue14', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue16', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue19', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue3', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue11', 900, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue12', 1300, 8, '100000', force_save=True)

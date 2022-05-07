import os
import argparse

from data.net2_data import net2_data

import warnings

warnings.filterwarnings("ignore")

if __name__ == '__main__':
    net2_data('data/superblue_0425_withHPWL/superblue6_processed', 811, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue7_processed', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue9_processed', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue14_processed', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue16_processed', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue19_processed', 700, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue3_processed', 800, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue11_processed', 900, 8, '100000', force_save=True)
    net2_data('data/superblue_0425_withHPWL/superblue12_processed', 1300, 8, '100000', force_save=True)

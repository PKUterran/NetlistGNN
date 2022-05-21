import os
import numpy as np
import pickle

from PIL import Image

CONG_FIG_DIR = 'cong'
if not os.path.isdir(CONG_FIG_DIR):
    os.mkdir(CONG_FIG_DIR)


def draw_cong(name, epoch=-1, fig_dir='temp'):
    pickle_path = f'{fig_dir}/{name}@{epoch}.pkl'
    if not os.path.exists(pickle_path):
        print(f'Not found: {name}@{epoch}')
        return
    with open(pickle_path, 'rb') as fp:
        x, y = pickle.load(fp)

    len_x, len_y = [160, 256]
    im1 = Image.new('RGB', (len_x, len_y))
    im2 = Image.new('RGB', (len_x, len_y))
    for xi in range(len_x):
        for yi in range(len_y):
            im1.putpixel((xi, yi), (
                0,
                int(25 ** x[xi, yi] * 10 - 10),
                0
            ))
            im2.putpixel((xi, yi), (
                0,
                int(25 ** y[xi, yi] * 10 - 10),
                0
            ))
    im1.save(f'{CONG_FIG_DIR}/{name}-target.png')
    im2.save(f'{CONG_FIG_DIR}/{name}.png')


def draw_cong_ganroute(name, epoch=-1, fig_dir='temp', out_name=None):
    pickle_path_ = f'{fig_dir}/hyper-test_@0.pkl'
    with open(pickle_path_, 'rb') as fp:
        x_, _ = pickle.load(fp)
    pickle_path = f'{fig_dir}/{name}@{epoch}.pkl'
    if not os.path.exists(pickle_path):
        print(f'Not found: {name}@{epoch}')
        return
    with open(pickle_path, 'rb') as fp:
        xs, ys = pickle.load(fp)
    
    indexes = {
        0: (0, 0),
        1: (1, 5),
        2: (2, 2),
        5: (3, 7),
        6: (4, 4),
        7: (0, 5),
        9: (0, 1),
        10: (1, 6),
        11: (2, 3),
        12: (2, 7),
        13: (3, 4),
        14: (4, 0),
        15: (4, 5),
        16: (0, 6),
        18: (1, 2),
        19: (1, 7),
        20: (0, 2),
        21: (3, 0),
        22: (3, 5),
        23: (4, 1),
        24: (4, 6),
        25: (0, 7),
        27: (1, 3),
        28: (2, 0),
        29: (2, 4),
        30: (3, 1),
        31: (0, 3),
        32: (4, 2),
        33: (4, 7),
        34: (1, 0),
        35: (2, 6),
        36: (1, 4),
        37: (2, 1),
        38: (2, 5),
        39: (3, 2),
        40: (3, 6),
        41: (4, 3),
        43: (1, 1),
        42: (0, 4),
        44: (3, 3),
    }
    x = np.zeros([160, 256])
    y = np.zeros([160, 256])
    for k, v in indexes.items():
        x[v[0] * 32: v[0] * 32 + 32, v[1] * 32: v[1] * 32 + 32] = xs[k]
        y[v[0] * 32: v[0] * 32 + 32, v[1] * 32: v[1] * 32 + 32] = ys[k]

    len_x, len_y = [160, 256]
    im1 = Image.new('RGB', (len_x, len_y))
    im2 = Image.new('RGB', (len_x, len_y))
    for xi in range(len_x):
        for yi in range(len_y):
            im1.putpixel((xi, yi), (
                0,
                int(25 ** x[xi, yi] * 28 - 28) if x_[xi, yi] > 0 else 0,
                0
            ))
            im2.putpixel((xi, yi), (
                0,
                int(25 ** y[xi, yi] * 28 - 28) if x_[xi, yi] > 0 else 0,
                0
            ))
    if name.startswith('hyper'):
        im1.save(f'{CONG_FIG_DIR}/cong-target.png')
    if out_name is None:
        out_name = name
    im2.save(f'{CONG_FIG_DIR}/{out_name}.png')


if __name__ == '__main__':
    draw_cong_ganroute('GanRoute-test_', 100)
    draw_cong('hyper-test_', 20)
    draw_cong('hyper-topo-test_', 20)
    draw_cong('LHNN-test_', 10)
    draw_cong('GAT-pos-test_', 20)

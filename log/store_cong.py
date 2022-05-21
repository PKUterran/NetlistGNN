import numpy as np
import pickle
# from matplotlib import pyplot as plt


def store_cong(target, source, name, epoch=-1, fig_dir='log/temp'):
    with open(f'{fig_dir}/{name}@{epoch}.pkl', 'wb+') as fp:
        pickle.dump((target, source), fp)


def store_cong_from_node(target, source, x, y, binx, biny, shape, name, epoch=-1, fig_dir='log/temp'):
    cmap_tgt = np.zeros(shape)
    cmap_prd = np.zeros_like(cmap_tgt)
    wmap = 1e-6 * np.ones_like(cmap_tgt)
    indices = []
    for i in range(0, target.shape[0]):
        key1, key2 = int(np.rint(x[i] / binx)), int(np.rint(y[i] / biny))
        if key1 == 0 and key2 == 0:
            continue
        wmap[key1][key2] += 1
        indices += [key2 + key1 * biny]
        cmap_prd[key1][key2] += source[i]
        cmap_tgt[key1][key2] += target[i]
    indices = list(set(indices))
    if 0 in indices:
        indices.remove(0)
    cmap_tgt = np.divide(cmap_tgt, wmap)
    cmap_prd = np.divide(cmap_prd, wmap)
    cmap_prd[0, 0] = 0
    cmap_tgt[0, 0] = 0

    with open(f'{fig_dir}/{name}@{epoch}.pkl', 'wb+') as fp:
        pickle.dump((cmap_tgt, cmap_prd), fp)


def store_cong_from_grid(target, source, part_x, part_y, shape, name, epoch=-1, fig_dir='log/temp'):
    x_size = int(shape[0] / part_x)
    y_size = int(shape[1] / part_y)
    tgt = np.zeros(shape)
    src = np.zeros(shape)
    for xi in range(x_size):
        for yi in range(y_size):
            for i in range(part_x):
                tgt[xi * part_x + i, yi * part_y: (yi + 1) * part_y] = target[
                    (xi * y_size + yi) * (part_x * part_y) + i * part_y:
                    (xi * y_size + yi) * (part_x * part_y) + (i + 1) * part_y
                ]
                src[xi * part_x + i, yi * part_y: (yi + 1) * part_y] = source[
                    (xi * y_size + yi) * (part_x * part_y) + i * part_y:
                    (xi * y_size + yi) * (part_x * part_y) + (i + 1) * part_y
                ]

    with open(f'{fig_dir}/{name}@{epoch}.pkl', 'wb+') as fp:
        pickle.dump((tgt, src), fp)

        
def store_cong_from_grid_ganroute(target, source, part_x, part_y, shape, name, epoch=-1, fig_dir='log/temp'):
    x_size = int(shape[0] / part_x)
    y_size = int(shape[1] / part_y)
    tgts, srcs = [], []
    for xi in range(x_size):
        for yi in range(y_size):
            tgt = np.zeros([part_x, part_y])
            src = np.zeros([part_x, part_y])
            for i in range(part_x):
                for j in range(part_y):
                    tgt[i, j] = target[
                        (xi + yi * x_size) * (part_x * part_y) + j * part_x + i
                    ]
                    src[i, j] = source[
                        (xi + yi * x_size) * (part_x * part_y) + j * part_x + i
                    ]
            tgts.append(tgt)
            srcs.append(src)

    with open(f'{fig_dir}/{name}@{epoch}.pkl', 'wb+') as fp:
        pickle.dump((tgts, srcs), fp)

import numpy as np
import torch
import torch.nn as nn

from PIL import Image
from typing import Tuple, List


def generate_rgb_init(h_capacity: np.ndarray, v_capacity: np.ndarray,
                      h_net_density: np.ndarray, v_net_density: np.ndarray,
                      pin_density: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    len_x = h_capacity.shape[0]
    len_y = h_capacity.shape[1]

    # h_capacity = np.zeros_like(h_capacity)
    # v_capacity = np.zeros_like(v_capacity)
    # h_net_density = np.zeros_like(h_net_density)
    # v_net_density = np.zeros_like(v_net_density)
    # pin_density = np.zeros_like(pin_density)

    r_channel = np.zeros(shape=[len_x * 2, len_y * 2], dtype=np.float)
    g_channel = np.zeros_like(r_channel)
    b_channel = np.zeros_like(r_channel)
    g_channel[1::2, ::2] = h_capacity + h_net_density
    b_channel[1::2, ::2] = h_capacity
    g_channel[::2, 1::2] = v_capacity + v_net_density
    b_channel[::2, 1::2] = v_capacity
    g_channel[1::2, 1::2] = h_net_density + v_net_density
    b_channel[1::2, 1::2] = pin_density
    return r_channel, g_channel * 64, b_channel * 6


def dump_data(dir_name: str, raw_dir_name: str, given_iter, bin_x: float = 32, bin_y: float = 40, use_tqdm=True
              ):
    h_capacity = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_capacity = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    h_net_density = np.load(f'{raw_dir_name}/hdm.npy')
    v_net_density = np.load(f'{raw_dir_name}/vdm.npy')
    xdata = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    ydata = np.load(f'{dir_name}/ydata_{given_iter}.npy')

    len_x = h_capacity.shape[0]
    len_y = h_capacity.shape[1]
    pin_density = np.zeros(shape=[len_x, len_y])
    for x, y in zip(xdata, ydata):
        if x < 1e-5 and y < 1e-5:
            continue
        key1 = int(np.rint(x / bin_x))
        key2 = int(np.rint(y / bin_y))
        if key1 >= len_x or key2 >= len_y:
            continue
        pin_density[key1, key2] += 1

    r_channel, g_channel, b_channel = generate_rgb_init(
        h_capacity=h_capacity, v_capacity=v_capacity,
        h_net_density=h_net_density, v_net_density=v_net_density,
        pin_density=pin_density
    )
    # print(len_x, len_y)
    # print(np.max(xdata) / bin_x, np.max(ydata) / bin_y)
    print(np.max(h_capacity))
    print(np.max(h_net_density))
    print(np.max(pin_density))
    print(np.max(r_channel))
    print(np.max(g_channel))
    print(np.max(b_channel))
    # print(pin_density)

    im = Image.new('RGB', (len_x, len_y))
    for x in range(len_x):
        for y in range(len_y):
            im.putpixel((x, y), (
                min(int(r_channel[x, y]), 255),
                min(int(g_channel[x, y]), 255),
                min(int(b_channel[x, y]), 255)
            ))
    im.save(f'{dir_name}/image.png')


def load_data(dir_name: str, given_iter, index: int, hashcode: str,
              graph_scale: int = 1000, bin_x: float = 40, bin_y: float = 32, force_save=False, use_tqdm=True
              ):
    pass

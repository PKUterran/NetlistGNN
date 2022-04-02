import os
import shutil
import numpy as np
import pickle
import torch
import torch.nn as nn
from torchvision import datasets, transforms

from PIL import Image
from typing import Tuple, List


INPUT_IMAGE = 'input'
OUTPUT_IMAGE = 'output'
EVAL_IMAGE = 'eval'


def generate_rgb_init(h_capacity: np.ndarray, v_capacity: np.ndarray,
                      h_net_density: np.ndarray, v_net_density: np.ndarray,
                      pin_density: np.ndarray, labels: np.ndarray, node_density: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    len_x = h_capacity.shape[0]
    len_y = h_capacity.shape[1]
    scale = 1

    # h_capacity = np.zeros_like(h_capacity)
    # v_capacity = np.zeros_like(v_capacity)
    # h_net_density = np.zeros_like(h_net_density)
    # v_net_density = np.zeros_like(v_net_density)
    # pin_density = np.zeros_like(pin_density)

    r_channel = np.zeros(shape=[len_x * 2, len_y * 2], dtype=np.float)
    g_channel = np.zeros_like(r_channel)
    b_channel = np.zeros_like(r_channel)
    grid_mask = np.zeros_like(r_channel)

    r_channel[::2, ::2] = labels
    r_channel[1::2, 1::2] = labels
    r_channel[1::2, ::2] = node_density
    g_channel[1::2, ::2] = h_capacity + h_net_density * scale
    b_channel[1::2, ::2] = h_capacity
    g_channel[::2, 1::2] = v_capacity + v_net_density * scale
    b_channel[::2, 1::2] = v_capacity
    g_channel[1::2, 1::2] = h_net_density * scale + v_net_density * scale
    b_channel[1::2, 1::2] = pin_density
    return r_channel * 180, g_channel * 64, b_channel * 2


def dump_data(dir_name: str, raw_dir_name: str, given_iter, index: int, hashcode: str,
              bin_x: float = 32, bin_y: float = 40, force_save=False, use_tqdm=True):
    print(f'processing {raw_dir_name}..')
    if not force_save and os.path.exists(f'{dir_name}/{INPUT_IMAGE}.png'):
        return

    with open(f'{dir_name}/edge.pkl', 'rb') as fp:
        edge = pickle.load(fp)
    h_capacity = np.load(f'{raw_dir_name}/hdm.npy')
    v_capacity = np.load(f'{raw_dir_name}/vdm.npy')
    h_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_h.npy')
    v_net_density = np.load(f'{raw_dir_name}/iter_{given_iter}_bad_cmap_v.npy')
    xdata = np.load(f'{dir_name}/xdata_{given_iter}.npy')
    ydata = np.load(f'{dir_name}/ydata_{given_iter}.npy')
    labels = np.load(f'{dir_name}/iter_{given_iter}_grid_label_full_{hashcode}_.npy')
    labels = labels[:, :, index]

    len_x = h_capacity.shape[0]
    len_y = h_capacity.shape[1]
    node_density = np.zeros(shape=[len_x, len_y])
    pin_density = np.zeros(shape=[len_x, len_y])
    for x, y in zip(xdata, ydata):
        if x < 1e-5 and y < 1e-5:
            continue
        key1 = int(np.rint(x / bin_x))
        key2 = int(np.rint(y / bin_y))
        if key1 >= len_x or key2 >= len_y:
            continue
        node_density[key1, key2] += 1
    for i, (_, list_node_feats) in enumerate(edge.items()):
        for node, pin_px, pin_py, _ in list_node_feats:
            px, py = xdata[node], ydata[node]
            if not px and not py:
                continue
            pin_density[int((px + pin_px) / bin_x), int((py + pin_py) / bin_y)] += 1

    r_channel, g_channel, b_channel = generate_rgb_init(
        h_capacity=h_capacity, v_capacity=v_capacity,
        h_net_density=h_net_density, v_net_density=v_net_density,
        pin_density=pin_density, labels=labels, node_density=node_density
    )
    # print(len_x, len_y)
    # print(np.max(xdata) / bin_x, np.max(ydata) / bin_y)
    print('\tmax h_capacity:', np.max(h_capacity))
    print('\tmax h_net_density:', np.max(h_net_density))
    print('\tmax pin_density:', np.max(pin_density))
    print('\tmax red:', np.max(r_channel[::2, ::2]))
    print('\tmax green:', np.max(g_channel))
    print('\tmax blue:', np.max(b_channel))
    print('\tnode density norm:', f'{np.sum(node_density > 0)} / {len_x * len_y}')
    print('\tmask:', f'{np.sum(r_channel[1::2, ::2] > 0)} / {len_x * len_y}')
    # print(pin_density)

    im1 = Image.new('RGB', (len_x, len_y))
    im2 = Image.new('RGB', (len_x, len_y))
    im3 = Image.new('RGB', (len_x, len_y))
    for x in range(len_x):
        for y in range(len_y):
            im1.putpixel((x, y), (
                0,
                min(int(g_channel[x, y]), 255),
                min(int(b_channel[x, y]), 255)
            ))
            im2.putpixel((x, y), (
                min(int(r_channel[x, y]), 255),
                min(int(g_channel[x, y]), 255),
                min(int(b_channel[x, y]), 255)
            ))
            im3.putpixel((x, y), (
                min(int(r_channel[x, y]), 255),
                0,
                0
            ))
    im1.save(f'{dir_name}/{INPUT_IMAGE}.png')
    im2.save(f'{dir_name}/{OUTPUT_IMAGE}.png')
    im3.save(f'{dir_name}/{EVAL_IMAGE}.png')


def collect_data(dirs: List[str], storage: str, part_x=64, part_y=64, clear_files=False):
    pres = [
        INPUT_IMAGE,
        OUTPUT_IMAGE,
        # EVAL_IMAGE,
    ]
    if not os.path.isdir(storage):
        os.mkdir(storage)
    for pre in pres:
        path = f'{storage}/{pre}'
        if os.path.isdir(path) and clear_files:
            shutil.rmtree(path)
        if not os.path.isdir(path):
            os.mkdir(path)
            os.mkdir(f'{path}/img')

    for pre in pres:
        cnt = 0
        for dir_name in dirs:
            im = Image.open(f'{dir_name}/{pre}.png')
            w, h = im.width, im.height
            max_x, max_y = int(w / part_x), int(h / part_y)
            for x in range(max_x):
                for y in range(max_y):
                    im_xy = im.crop((x * part_x, y * part_y, (x + 1) * part_x, (y + 1) * part_y))
                    im_xy.save(f'{storage}/{pre}/img/{cnt}.png')
                    cnt += 1


def load_data(storage: str) -> Tuple[datasets.ImageFolder, datasets.ImageFolder]:
    data_transform = transforms.Compose([
        # transforms.Resize((300, 300)),
        transforms.ToTensor(),
    ])
    input_images = datasets.ImageFolder(f'{storage}/{INPUT_IMAGE}', transform=data_transform)
    output_images = datasets.ImageFolder(f'{storage}/{OUTPUT_IMAGE}', transform=data_transform)
    return input_images, output_images

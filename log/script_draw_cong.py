import os
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

    len_x, len_y = x.shape
    im1 = Image.new('RGB', (len_x, len_y))
    im2 = Image.new('RGB', (len_x, len_y))
    for xi in range(len_x):
        for yi in range(len_y):
            im1.putpixel((xi, yi), (
                0,
                x[xi, yi],
                0
            ))
            im2.putpixel((xi, yi), (
                0,
                y[xi, yi],
                0
            ))
    im1.save(f'{CONG_FIG_DIR}/{name}-target.png')
    im2.save(f'{CONG_FIG_DIR}/{name}.png')


if __name__ == '__main__':
    draw_cong()

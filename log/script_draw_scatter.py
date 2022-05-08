import os
import pickle
from matplotlib import pyplot as plt

HPWL_FIG_DIR = 'hpwl-scatter'
if not os.path.isdir(HPWL_FIG_DIR):
    os.mkdir(HPWL_FIG_DIR)


def draw_scatter_hpwl(name, epoch=-1, fig_dir='hpwl-temp'):
    pickle_path = f'{fig_dir}/{name}@{epoch}.pkl'
    if not os.path.exists(pickle_path):
        print(f'Not found: {name}@{epoch}')
        return
    with open(pickle_path, 'rb') as fp:
        x, y = pickle.load(fp)
    fig = plt.figure(figsize=(3, 3))
    
    s_x = sorted(x)
    s_y = sorted(y)
    n = len(s_x)
    min_x = s_x[int(n * 0.002)] - 1.0
    max_x = s_x[int(n * 0.998)] + 1.0
    min_y = s_y[int(n * 0.002)] - 0.1
    max_y = s_y[int(n * 0.998)] + 0.1
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.plot([min_x, max_x], [min_y, max_y], linestyle='--', color='red')
    plt.xticks([])
    plt.yticks([])
    
#     lim = 4
#     plt.xlim(-lim, lim)
#     plt.ylim(-lim, lim)
#     plt.plot([-lim, lim], [-lim, lim], linestyle='--', color='red')

    plt.scatter(x, y, s=1)
    plt.savefig(f'{HPWL_FIG_DIR}/{name}.png')


if __name__ == '__main__':
    draw_scatter_hpwl('MLP-test_', 20)
    draw_scatter_hpwl('Net2f-test_', 13)
    draw_scatter_hpwl('Net2a-test_', 10)
    draw_scatter_hpwl('LHNN-test_', 10)
    draw_scatter_hpwl('hyper-test_', 11)

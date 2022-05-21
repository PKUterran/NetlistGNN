import os
import pickle
from matplotlib import pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

HPWL_FIG_DIR = 'hpwl-scatter'
if not os.path.isdir(HPWL_FIG_DIR):
    os.mkdir(HPWL_FIG_DIR)


def draw_scatter_hpwl(name, epoch=-1, fig_dir='hpwl-temp', out_name=None):
    pickle_path = f'{fig_dir}/{name}@{epoch}.pkl'
    if not os.path.exists(pickle_path):
        print(f'Not found: {name}@{epoch}')
        return
    with open(pickle_path, 'rb') as fp:
        x, y = pickle.load(fp)
        
    fig = plt.figure(figsize=(3, 3))
    
    if name.startswith('LHNN'):
        x, y = x[::522], y[::522]
    else:
        x, y = x[::3000], y[::3000]
    print(len(x))
    s_x = sorted(x)
    s_y = sorted(y)
    n = len(s_x)
    min_x = s_x[int(n * 0.001)] - 0.2
    max_x = s_x[int(n * 0.999)] + 0.2
    min_y = s_y[int(n * 0.001)] - 0.02
    max_y = s_y[int(n * 0.999)] + 0.02
    plt.xlim(min_x, max_x)
    plt.ylim(min_y, max_y)
    plt.xticks([])
    plt.yticks([])
    
#     lim = 4
#     plt.xlim(-lim, lim)
#     plt.ylim(-lim, lim)
#     plt.plot([-lim, lim], [-lim, lim], linestyle='--', color='red')

#     plt.plot([min_x, max_x], [min_y, max_y], linestyle='--', color='red')
#     plt.scatter(x, y, s=1)

#     fig = plt.figure(figsize=(8, 8))
    sns.regplot(x, y, color='#66B2FF')
    
    if out_name is None:
        out_name = name
    plt.savefig(f'{HPWL_FIG_DIR}/{out_name}.png')


if __name__ == '__main__':
    draw_scatter_hpwl('MLP-test_', 20, out_name='hpwl-MLP')
    draw_scatter_hpwl('Net2f-test_', 13, out_name='hpwl-Net2f')
    draw_scatter_hpwl('Net2a-test_', 10, out_name='hpwl-Net2a')
    draw_scatter_hpwl('LHNN-test_', 10, out_name='hpwl-LHNN')
    draw_scatter_hpwl('hyper-test_', 11, out_name='hpwl-ours')

import pickle
# from matplotlib import pyplot as plt


def draw_scatter(x, y, name, epoch=-1, fig_dir='log/temp'):
    with open(f'{fig_dir}/{name}@{epoch}.pkl', 'wb+') as fp:
        pickle.dump((x, y), fp)
#     fig = plt.figure(figsize=(8, 8))
#     plt.scatter(x, y, s=1)
#     plt.savefig(f'{fig_dir}/{name}.png')

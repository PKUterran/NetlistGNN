from matplotlib import pyplot as plt


def draw_scatter(x, y, name, fig_dir='log/temp'):
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(x, y, s=1)
    plt.savefig(f'{fig_dir}/{name}.png')

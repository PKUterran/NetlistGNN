import numpy as np
import pickle


DATA_DIR = 'data/superblue19_processed'

with open(f'{DATA_DIR}/edg.pkl', 'rb') as fp:
    edg = pickle.load(fp)
with open(f'{DATA_DIR}/nodef.pkl', 'rb') as fp:
    nodef = pickle.load(fp)
print(type(edg))
print(len(edg.keys()))
print(edg[0])
print(sum([len(v) for v in edg.values()]))
print(type(nodef))
print(len(nodef.keys()))
print(nodef[0])

grid_label = np.load(f'{DATA_DIR}/iter_900_grid_label_full_000000_.npy')
node_label = np.load(f'{DATA_DIR}/iter_900_node_label_full_000000_.npy')
print(grid_label.shape)
print(grid_label[12, 34])
print(node_label.shape)
print(node_label[0])

pos_x = np.load(f'{DATA_DIR}/pos_900_xdata.npy')
pos_y = np.load(f'{DATA_DIR}/pos_900_ydata.npy')
size_x = np.load(f'{DATA_DIR}/sizdata_x.npy')
size_y = np.load(f'{DATA_DIR}/sizdata_y.npy')
pdata = np.load(f'{DATA_DIR}/pdata.npy')
print(pos_x.shape)
print(pos_x[:5])
print(pos_y.shape)
print(pos_y[:5])
print(size_x.shape)
print(size_x[:5])
print(size_y.shape)
print(size_y[:5])
print(pdata.shape)
print(pdata[:5])

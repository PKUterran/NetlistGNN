import numpy as np
import pickle


DATA_DIR = 'data/superblue16_processed'

with open(f'{DATA_DIR}/edge.pkl', 'rb') as fp:
    edge = pickle.load(fp)
with open(f'{DATA_DIR}/nodef.pkl', 'rb') as fp:
    nodef = pickle.load(fp)
print(type(edge))
print(len(edge.keys()))
print(edge[0])
t = edge[0][0][0]
# print(sum([len(v) for v in edge.values()]))
print(type(nodef))
print(len(nodef.keys()))
print(nodef[t])

# grid_label = np.load(f'{DATA_DIR}/iter_700_grid_label_full_000000_.npy')
node_label = np.load(f'{DATA_DIR}/iter_700_node_label_full_000000_.npy')
# print(grid_label.shape)
# print(grid_label[12, 34])
print(node_label.shape)
print(node_label[0])
#
pos_x = np.load(f'{DATA_DIR}/xdata_700.npy')
pos_y = np.load(f'{DATA_DIR}/ydata_700.npy')
size_x = np.load(f'{DATA_DIR}/sizdata_x.npy')
size_y = np.load(f'{DATA_DIR}/sizdata_y.npy')
pdata = np.load(f'{DATA_DIR}/pdata.npy')
print(pos_x.shape)
print(np.min(pos_x))
print(pos_y.shape)
print(np.min(pos_y))
print(size_x.shape)
# print(size_x[:5])
print(size_y.shape)
# print(size_y[:5])
print(pdata.shape)
# print(pdata[:5])

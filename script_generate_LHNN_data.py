from data.LHNN_data import dump_data as dump_data_lhnn
from data.LHNN_data import load_data as load_data_lhnn


if __name__ == '__main__':
    dump_data_lhnn('data/superblue7_processed', 'data/superblue7', 800, 8, '100000', force_save=True)
    dump_data_lhnn('data/superblue9_processed', 'data/superblue9', 800, 8, '100000', force_save=True)
    dump_data_lhnn('data/superblue14_processed', 'data/superblue14', 800, 8, '100000', force_save=True)
    dump_data_lhnn('data/superblue16_processed', 'data/superblue16', 700, 8, '100000', force_save=True)
    dump_data_lhnn('data/superblue19_processed', 'data/superblue19', 900, 8, '100000', force_save=True)
    list_tensors = load_data_lhnn('data/superblue7_processed', 800)
    print(len(list_tensors))
    for list_tensor in list_tensors:
        print([tensor.shape for tensor in list_tensor])
    print(list_tensors[0])
    for u, vs in list_tensors[125][3].link_set.items():
        print(len(vs))

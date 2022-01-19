import numpy as np
import os
import sys
import pickle
import argparse
from process_iter_data import proc_iter_data


def convertpin(givenstr):
    if givenstr.decode() == 'INPUT':
        return 0
    else:
        return 1


def processintofour(twodmap, caps):
    unnorm_map = np.multiply(twodmap, caps.reshape(1, 1, -1))
    hv_max = np.amax(twodmap, axis=2, keepdims=True)
    hv_max_sum_cap = np.amax(unnorm_map, axis=2, keepdims=True) / np.sum(caps)
    hv_mean = twodmap.mean(axis=2, keepdims=True)
    hv_mean_sum_cap = unnorm_map.mean(axis=2, keepdims=True) / np.sum(caps)
    return np.concatenate((hv_max, hv_max_sum_cap, hv_mean, hv_mean_sum_cap), axis=2)


## INEFFICIENT, needs some vectorization

def normalize3darray(arr, mask):
    for z in range(arr.shape[2]):
        arr[:, :, z] = normalize2darray(arr[:, :, z], mask[:, :, z])
    return arr


def normalize2darray(arr, mask):
    cnt = np.sum(mask)
    arr_mean = np.sum(arr) / cnt
    temp_hold = np.sum(np.multiply(arr, arr)) / cnt
    var = temp_hold - arr_mean * arr_mean
    sd = np.sqrt(var)
    arr = (arr - arr_mean) / sd
    return np.multiply(arr, mask)


def loadmanyarrays(listofnames, commonprefix):
    return [np.load(commonprefix + listofnames[i] + '.npy') for i in range(len(listofnames))]


def remapsomearrays(twodarr, mapper, sourcearrays, threshold=1e9):
    for j in range(len(sourcearrays)):
        for i in range(len(sourcearrays[j])):
            if (i >= threshold):
                break
            twodarr[j][mapper[i]] = sourcearrays[j][i]


def savemanyarrays(listofarrays, listofnames, commonprefix):
    [np.save(commonprefix + listofnames[i] + '.npy', np.transpose(listofarrays[i])) for i in range(len(listofarrays))]
    return


def process(datadir, binx, biny, cong, nctu_flag, mask_flag, init_flag, norm_flag, zeromean_flag, lower_flag):
    flag = True
    outdir = datadir + "_processed"

    for z in range(1, 3000):
        if (not (os.path.isfile(datadir + '/iter_' + str(z) + '_x.npy'))):
            continue
        if not os.path.exists(outdir):
            os.makedirs(outdir)

        if (cong):
            processcongestion(datadir, binx, biny, nctu_flag, mask_flag, init_flag, norm_flag, z, zeromean_flag,
                              lower_flag)
            continue

        mapper = np.load(datadir + '/iter_fix_mapper.npy')

        upper, lower = np.max(mapper), np.min(mapper)

        pin_and_other = loadmanyarrays(['po_x', 'po_y', 'pd'], datadir + '/iter_fix_')
        pinbasedfeats, endict = {}, {}

        pincnt = np.zeros(upper - lower + 1)

        pinstuff = loadmanyarrays(['pin2edge', 'pin2node'], datadir + '/iter_fix_')

        for i in range(0, len(pinstuff[0])):
            endict.setdefault(pinstuff[0][i], []).append(mapper[pinstuff[1][i]])

        for i in range(0, len(pinstuff[1])):
            pincnt[mapper[pinstuff[1][i]]] += 1

        for i in range(0, len(pinstuff[1])):
            pinbasedfeats.setdefault(mapper[pinstuff[1][i]]).append((
                    pinstuff[0][i],
                    pin_and_other[0][i], pin_and_other[1][i], convertpin(pin_and_other[2][i])
                ))

        position = loadmanyarrays(['_x', '_y'], datadir + '/iter_' + str(z))
        sizes = loadmanyarrays(['_sizes_x', '_sizes_y'], datadir + '/iter_' + 'fix')

        final_position_and_sizes = np.zeros((4, upper - lower + 1))

        remapsomearrays(final_position_and_sizes, mapper, position + sizes, threshold=len(position[0]))

        if (flag):
            with open(outdir + '/edg.pkl', 'wb') as fp:
                pickle.dump(endict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open(outdir + '/nodef' + '.pkl', 'wb') as fp:
                pickle.dump(pinbasedfeats, fp, protocol=pickle.HIGHEST_PROTOCOL)
        if (flag):
            np.save(outdir + '/pdata' + '.npy', pincnt)
            savemanyarrays(final_position_and_sizes[2:, :], ['sizdata_x', 'sizdata_y'], outdir + '/')
            flag = False

        savemanyarrays(final_position_and_sizes[:2, :], ['_xdata', '_ydata'], outdir + '/pos_' + str(z))


def fixinitial(basearr, initarr):
    return (basearr - np.repeat(np.expand_dims(initarr, axis=2), basearr.shape[2], axis=2))


def processcongestion(datadir, binx, biny, nctu_flag, mask_flag, init_flag, norm_flag, z, zeromean_flag, lower_flag):
    congprefix = '' if nctu_flag else '_bad'
    outdir = datadir + "_processed"
    fixprefix = datadir + '/iter_fix'
    variableprefix = datadir + '/iter_' + str(z)

    processed_designation = str(int(nctu_flag)) + str(int(mask_flag)) + str(int(init_flag)) + str(int(norm_flag)) + str(
        int(zeromean_flag)) + str(int(lower_flag))

    listofnames = ['_x', '_y', congprefix + '_cmap_h', congprefix + '_cmap_v']
    pos_and_cmap = loadmanyarrays(listofnames, variableprefix)

    cmap_h, cmap_v = pos_and_cmap[-2], pos_and_cmap[-1]
    if not nctu_flag:
        cmap_h = np.expand_dims(cmap_h, axis=2)
        cmap_v = np.expand_dims(cmap_v, axis=2)
    x, y = pos_and_cmap[0], pos_and_cmap[1]

    if (init_flag):
        initcong = loadmanyarrays(['/hdm', '/vdm'], datadir)
        cmap_h, cmap_v = fixinitial(cmap_h, initcong[0]), fixinitial(cmap_v, initcong[1])

    basal_cmap = np.maximum(cmap_h, cmap_v)

    if (nctu_flag):
        cap_nctu = np.load(fixprefix + '_nctu_cap.npy').flatten()
        caps_h, caps_v = cap_nctu[2::2], cap_nctu[1::2]
        if (lower_flag):
            caps_h, caps_v = caps_h[:2], caps_v[:2]
            cmap_h, cmap_v = cmap_h[:, :, :2], cmap_v[:, :, :2]
        remadecaps = np.concatenate((caps_h, caps_v))
        horiz = processintofour(cmap_h, caps_h)
        vert = processintofour(cmap_v, caps_v)
    else:
        horiz = np.repeat(cmap_h, 4, axis=2)
        vert = np.repeat(cmap_v, 4, axis=2)
        remadecaps = np.asarray(loadmanyarrays(['_hori_cap', '_verti_cap'], fixprefix))
    overall_cmap = np.concatenate((cmap_h, cmap_v), axis=2)
    overall_final_cmap = processintofour(overall_cmap, remadecaps)
    full_label = np.concatenate((horiz, vert, overall_final_cmap), axis=2)

    if (mask_flag or norm_flag or zeromean_flag):
        count_map = 1e-9 * np.ones_like(full_label)
        for i in range(0, len(x)):
            key1, key2 = int(np.rint(x[i] / binx)), int(np.rint(y[i] / biny))
            count_map[key1, key2, :] += 1
        mask_map = np.clip(count_map, 0, 1)
        mask_map[0, 0, :] = 0
        full_label = np.multiply(full_label, mask_map)
        if (norm_flag):
            full_label = np.divide(full_label, count_map)
        if (zeromean_flag):
            full_label = normalize3darray(full_label, mask_map)

    ## THIS IS THE GRID-LEVEL CONGESTION

    np.save(outdir + '/iter_' + str(z) + '_grid_label_full_' + processed_designation + '_' + '.npy', full_label)

    mapper = np.load(fixprefix + '_mapper.npy')
    node_level_labels = np.zeros((np.max(mapper) - np.min(mapper), 12))

    # GRID LEVEL SAVE

    for i in range(0, len(x)):
        key1, key2 = int(np.rint(x[i] / binx)), int(np.rint(y[i] / biny))
        node_level_labels[mapper[i], :] = full_label[key1, key2, :]

    ## THIS IS THE NODE-LEVEL CONGESTION

    np.save(outdir + '/iter_' + str(z) + '_node_label_full_' + processed_designation + '_' + '.npy', node_level_labels)


if __name__ == '__main__':
    '''
    This file processes the congestion map or non-congestion parts.
    Based on the arguments congestion will be either RUDY/NCTU, masked/unmasked, initial-adjusted/not, normalized/not
    '''
    argparser = argparse.ArgumentParser("Training", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    argparser.add_argument('--datadir', type=str, default='superblue7', help="Data directory.")
    argparser.add_argument('--binx', type=int, default=32, help="Number of bins in x direction.")
    argparser.add_argument('--biny', type=str, default=40, help="Number of bins in y direction.")
    argparser.add_argument('--cong', type=bool, default=True, help="Whether generate a congestion map.")
    argparser.add_argument('--nctu', type=bool, default=False, help="RUDY will be used if this is set to False.")
    argparser.add_argument('--mask', type=bool, default=False,
                           help="The grid level congestion is zeroed when no node is present.")
    argparser.add_argument('--initial', type=bool, default=False, help="Inital congestion is subtracted.")
    argparser.add_argument('--norm', type=bool, default=False, help="Raw value is divided by number of nodes instead.")
    argparser.add_argument('--setzeromean', type=bool, default=False, help="Set zero mean and unit variance.")
    argparser.add_argument('--lower', type=bool, default=False, help="Extract only lower level congestion.")

    args = argparser.parse_args()
    process(args.datadir, args.binx, args.biny, args.cong, args.nctu, args.mask, args.initial, args.norm,
            args.setzeromean, args.lower)
    proc_iter_data(args.datadir)

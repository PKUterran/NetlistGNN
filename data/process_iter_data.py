import numpy as np
import os
import sys
from scipy.stats import pearsonr
from scipy.stats import pearsonr, spearmanr, kendalltau


def printout(arr1, arr2, prefix=""):
    print(prefix + "pearson", pearsonr(arr1, arr2))
    print(prefix + "spearman", spearmanr(arr1, arr2))
    print(prefix + "kendall", kendalltau(arr1, arr2))
    print(prefix + "MAE", np.sum(np.abs(arr1 - arr2)) / len(arr1))
    return


def map_back(val, lo, high, binsize, maxbin):
    if (val < lo):
        return 0
    if (val > high):
        return maxbin
    return (int(np.floor((val - lo) / binsize)))


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


def normalize2darray(arr, mask):
    cnt = np.sum(mask)
    arr_mean = np.sum(arr) / cnt
    temp_hold = np.sum(np.multiply(arr, arr)) / cnt
    var = temp_hold - arr_mean * arr_mean
    sd = np.sqrt(var)
    arr = (arr - arr_mean) / sd
    return np.multiply(arr, mask)


def proc_iter_data(datadir):
    flag = True
    outdir = datadir + "_processed"
    for z in range(1, 3000):
        if (not (os.path.isfile(datadir + '/iter_' + str(z) + '_x.npy'))):
            continue
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        x = np.load(datadir + '/iter_' + str(z) + '_x.npy')
        y = np.load(datadir + '/iter_' + str(z) + '_y.npy')

        '''
        binx = 20.8828
        biny = 20.8594

        xl = 459
        yl = 459

        xh,yh = 11151, 11139
        '''

        binx = 32
        biny = 40

        rudy_cmap_h = np.load(datadir + '/iter_' + str(z) + '_bad_cmap_h.npy')
        rudy_cmap_v = np.load(datadir + '/iter_' + str(z) + '_bad_cmap_v.npy')

        print(f"RUDY: Max rudy_cmap_h: {np.max(rudy_cmap_h)}")
        print(f"RUDY: Min rudy_cmap_h: {np.min(rudy_cmap_h)}")

        cmap_h = np.load(datadir + '/iter_' + str(z) + '_cmap_h.npy')
        cmap_v = np.load(datadir + '/iter_' + str(z) + '_cmap_v.npy')

        init_h = np.load(datadir + '/hdm.npy')
        init_v = np.load(datadir + '/vdm.npy')

        rudy_cmap_h_normed = rudy_cmap_h - init_h
        rudy_cmap_v_normed = rudy_cmap_v - init_v

        cmap_h_normed = np.zeros_like(cmap_h)
        cmap_v_normed = np.zeros_like(cmap_v)

        for i in range(0, cmap_h.shape[2]):
            cmap_h_normed[:, :, i] = cmap_h[:, :, i] - init_h
            cmap_v_normed[:, :, i] = cmap_v[:, :, i] - init_v

        print(f"NCTU: Max cmap_v: {np.max(cmap_v)}")
        print(f"NCTU: Min cmap_v: {np.min(cmap_v)}")
        print(f"NCTU: Max cmap_h: {np.max(cmap_h)}")
        print(f"NCTU: Min cmap_h: {np.min(cmap_h)}")
        print(f"NCTU: Max cmap_v_normed: {np.max(cmap_v_normed)}")
        print(f"NCTU: Min cmap_v_normed: {np.min(cmap_v_normed)}")
        print(f"NCTU: Max cmap_h_normed: {np.max(cmap_h_normed)}")
        print(f"NCTU: Min cmap_h_normed: {np.min(cmap_h_normed)}")

        print(f"NCTU: shape of cmap_h: {np.shape(cmap_h)}")

        cmap = np.maximum(cmap_h, cmap_v)
        cmap_norm = np.maximum(cmap_h_normed, cmap_v_normed)

        rudy_cmap = np.maximum(rudy_cmap_h, rudy_cmap_v)
        rudy_cmap_norm = np.maximum(rudy_cmap_h_normed, rudy_cmap_v_normed)

        full_nct_label = np.zeros((cmap_h.shape[0], cmap_h.shape[1], 12))
        full_rudy_label = np.zeros_like(full_nct_label)

        nctu_labels = np.zeros((len(x), 24))
        rudy_labels = np.zeros((len(x), 24))

        nctu_mask_labels = np.zeros((len(x), 24))
        rudy_mask_labels = np.zeros((len(x), 24))

        nctu_norm_labels = np.zeros((len(x), 24))
        rudy_norm_labels = np.zeros((len(x), 24))

        nctu_norm_mask_labels = np.zeros((len(x), 24))
        rudy_norm_mask_labels = np.zeros((len(x), 24))

        hori_rudy = np.load(datadir + '/iter_' + str('fix') + '_hori_cap.npy')
        ver_rudy = np.load(datadir + '/iter_' + str('fix') + '_verti_cap.npy')
        cap_rudy = np.asarray([hori_rudy, ver_rudy])

        print(f"Shape of cap_rudy: {np.shape(cap_rudy)}")

        cap_nctu = np.load(datadir + '/iter_' + str('fix') + '_nctu_cap.npy').flatten()

        print(f"Shape of cap_nctu: {np.shape(cap_nctu)}")

        caps_h = cap_nctu[2::2]
        caps_v = cap_nctu[1::2]

        remadecaps = np.concatenate((caps_h, caps_v))

        horiz = processintofour(cmap_h, caps_h)
        vert = processintofour(cmap_v, caps_v)
        overall_nctu_cmap = np.concatenate((cmap_h, cmap_v), axis=2)
        overall_nct = processintofour(overall_nctu_cmap, remadecaps)

        full_nct_label = np.concatenate((horiz, vert, overall_nct), axis=2)

        rudy_cmap_h = np.expand_dims(rudy_cmap_h, axis=2)
        rudy_cmap_v = np.expand_dims(rudy_cmap_v, axis=2)

        rudy_cmap_h_normed = np.expand_dims(rudy_cmap_h_normed, axis=2)
        rudy_cmap_v_normed = np.expand_dims(rudy_cmap_v_normed, axis=2)

        horiz_rudy_label = np.repeat(rudy_cmap_h, 4, axis=2)
        vert_rudy_label = np.repeat(rudy_cmap_v, 4, axis=2)
        altcmap_o = np.concatenate((rudy_cmap_h, rudy_cmap_v), axis=2)
        overall_rudy_label = processintofour(altcmap_o, cap_rudy)
        full_rudy_label = np.concatenate((horiz_rudy_label, vert_rudy_label, overall_rudy_label), axis=2)

        horiz_normed = processintofour(cmap_h_normed, caps_h)
        vert_normed = processintofour(cmap_v_normed, caps_v)
        overall_nctu_cmap_normed = np.concatenate((cmap_h_normed, cmap_v_normed), axis=2)
        overall_nct_normed = processintofour(overall_nctu_cmap_normed, remadecaps)

        full_nct_label_normed = np.concatenate((horiz_normed, vert_normed, overall_nct_normed), axis=2)

        horiz_rudy_label_normed = np.repeat(rudy_cmap_h_normed, 4, axis=2)
        vert_rudy_label_normed = np.repeat(rudy_cmap_v_normed, 4, axis=2)
        altcmap_o_normed = np.concatenate((rudy_cmap_h_normed, rudy_cmap_v_normed), axis=2)
        overall_rudy_label_normed = processintofour(altcmap_o_normed, cap_rudy)
        full_rudy_label_normed = np.concatenate(
            (horiz_rudy_label_normed, vert_rudy_label_normed, overall_rudy_label_normed), axis=2)

        np.save(outdir + '/iter_' + str(z) + '_correct_label_full.npy', full_nct_label)
        np.save(outdir + '/iter_' + str(z) + '_bad_label_full.npy', full_rudy_label)

        np.save(outdir + '/iter_' + str(z) + '_correct_label_full_normed.npy', full_nct_label_normed)
        np.save(outdir + '/iter_' + str(z) + '_bad_label_full_normed.npy', full_rudy_label_normed)

        wmap = np.ones_like(full_nct_label)
        masktouse = np.zeros_like(full_nct_label)

        indices = []

        for i in range(0, len(x)):
            key1 = int(np.rint(x[i] / binx))
            key2 = int(np.rint(y[i] / biny))
            wmap[key1, key2, :] += 1
            masktouse[key1, key2, :] = 1

        masktouse[0, 0, :] = 0

        masked_nctu_map_orig = np.multiply(full_nct_label, masktouse)
        masked_rudy_map_orig = np.multiply(full_rudy_label, masktouse)

        masked_nctu_map_orig_normed = np.multiply(full_nct_label_normed, masktouse)
        masked_rudy_map_orig_normed = np.multiply(full_rudy_label_normed, masktouse)

        masked_nctu_map = np.zeros_like(masked_nctu_map_orig)
        masked_rudy_map = np.zeros_like(masked_rudy_map_orig)

        masked_nctu_map_normed = np.zeros_like(masked_nctu_map_orig_normed)
        masked_rudy_map_normed = np.zeros_like(masked_rudy_map_orig_normed)

        for j in range(0, 12):
            masked_nctu_map[:, :, j] = normalize2darray(masked_nctu_map_orig[:, :, j], masktouse[:, :, j])
            masked_rudy_map[:, :, j] = normalize2darray(masked_rudy_map_orig[:, :, j], masktouse[:, :, j])
            masked_rudy_map_normed[:, :, j] = normalize2darray(masked_rudy_map_orig_normed[:, :, j], masktouse[:, :, j])
            masked_nctu_map_normed[:, :, j] = normalize2darray(masked_nctu_map_orig_normed[:, :, j], masktouse[:, :, j])
        np.save(outdir + '/iter_' + str(z) + '_correct_label_full_masked.npy', masked_nctu_map)
        np.save(outdir + '/iter_' + str(z) + '_bad_label_full_masked.npy', masked_rudy_map)

        np.save(outdir + '/iter_' + str(z) + '_correct_label_full_masked_normed.npy', masked_nctu_map_normed)
        np.save(outdir + '/iter_' + str(z) + '_bad_label_full_masked_normed.npy', masked_rudy_map_normed)

        normalized_nct = np.divide(full_nct_label, wmap)
        normalized_rudy = np.divide(full_rudy_label, wmap)

        normalized_nct_mask = np.divide(masked_nctu_map, wmap)
        normalized_rudy_mask = np.divide(masked_rudy_map, wmap)

        normalized_nct_normed = np.divide(full_nct_label_normed, wmap)
        normalized_rudy_normed = np.divide(full_rudy_label_normed, wmap)

        normalized_nct_mask_normed = np.divide(masked_nctu_map_normed, wmap)
        normalized_rudy_mask_normed = np.divide(masked_rudy_map_normed, wmap)

        for i in range(0, len(x)):
            key1 = int(np.rint(x[i] / binx))
            key2 = int(np.rint(y[i] / biny))
            nctu_labels[i, :12] = normalized_nct[key1, key2, :]
            rudy_labels[i, :12] = normalized_rudy[key1, key2, :]

            nctu_labels[i, 12:] = full_nct_label[key1, key2, :]
            rudy_labels[i, 12:] = full_rudy_label[key1, key2, :]

            nctu_mask_labels[i, :12] = normalized_nct_mask[key1, key2, :]
            rudy_mask_labels[i, :12] = normalized_rudy_mask[key1, key2, :]

            nctu_mask_labels[i, 12:] = masked_nctu_map[key1, key2, :]
            rudy_mask_labels[i, 12:] = masked_rudy_map[key1, key2, :]

            nctu_norm_labels[i, :12] = normalized_nct_normed[key1, key2, :]
            rudy_norm_labels[i, :12] = normalized_rudy_normed[key1, key2, :]

            nctu_norm_labels[i, 12:] = full_nct_label_normed[key1, key2, :]
            rudy_norm_labels[i, 12:] = full_rudy_label_normed[key1, key2, :]

            nctu_norm_mask_labels[i, :12] = normalized_nct_mask_normed[key1, key2, :]
            rudy_norm_mask_labels[i, :12] = normalized_rudy_mask_normed[key1, key2, :]

            nctu_norm_mask_labels[i, 12:] = masked_nctu_map_normed[key1, key2, :]
            rudy_norm_mask_labels[i, 12:] = masked_rudy_map_normed[key1, key2, :]

        mapper = np.load(datadir + '/iter_' + str('fix') + '_mapper.npy')

        take2 = np.max(mapper)
        take1 = np.min(mapper)

        truecon = np.zeros((take2 - take1 + 1, 24))
        fakecon = np.zeros((take2 - take1 + 1, 24))

        truecon_mask = np.zeros((take2 - take1 + 1, 24))
        fakecon_mask = np.zeros((take2 - take1 + 1, 24))

        truecon_norm = np.zeros((take2 - take1 + 1, 24))
        fakecon_norm = np.zeros((take2 - take1 + 1, 24))

        truecon_mask_norm = np.zeros((take2 - take1 + 1, 24))
        fakecon_mask_norm = np.zeros((take2 - take1 + 1, 24))

        truex = np.zeros(take2 - take1 + 1)
        truey = np.zeros(take2 - take1 + 1)
        trusize_x = np.zeros(take2 - take1 + 1)
        trusize_y = np.zeros(take2 - take1 + 1)
        pincnt = np.zeros(take2 - take1 + 1)

        pinbasedfeats = dict()

        sizearr_x = np.load(datadir + '/iter_' + str('fix') + '_sizes_x.npy')
        sizearr_y = np.load(datadir + '/iter_' + str('fix') + '_sizes_y.npy')

        pinarr_x = np.load(datadir + '/iter_' + str('fix') + '_po_x.npy')
        pinarr_y = np.load(datadir + '/iter_' + str('fix') + '_po_y.npy')
        pd = np.load(datadir + '/iter_' + str('fix') + '_pd.npy')

        for i in range(0, len(x)):
            truex[mapper[i]] = x[i]
            truey[mapper[i]] = y[i]
            truecon[mapper[i], :] = nctu_labels[i, :]
            fakecon[mapper[i], :] = rudy_labels[i, :]

            truecon_mask[mapper[i], :] = nctu_mask_labels[i, :]
            fakecon_mask[mapper[i], :] = rudy_mask_labels[i, :]

            truecon_norm[mapper[i], :] = nctu_norm_labels[i, :]
            fakecon_norm[mapper[i], :] = rudy_norm_labels[i, :]

            truecon_mask_norm[mapper[i], :] = nctu_norm_mask_labels[i, :]
            fakecon_mask_norm[mapper[i], :] = rudy_norm_mask_labels[i, :]

            trusize_x[mapper[i]] = sizearr_x[i]
            trusize_y[mapper[i]] = sizearr_y[i]

        pin1 = np.load(datadir + '/iter_' + str('fix') + '_pin2edge.npy')
        pin2 = np.load(datadir + '/iter_' + str('fix') + '_pin2node.npy')

        endict = {}
        edge_dict = {}

        for i in range(0, len(pin1)):
            endict.setdefault(pin1[i], []).append(mapper[pin2[i]])
            edge_dict.setdefault(pin1[i], []).append((mapper[pin2[i]],
                                                      pinarr_x[i], pinarr_y[i], convertpin(pd[i])))

        for i in range(0, len(pin2)):
            pincnt[mapper[pin2[i]]] += 1

        for i in range(0, len(pin2)):
            pinbasedfeats.setdefault(mapper[pin2[i]], []).append((pinarr_x[i], pinarr_y[i], convertpin(pd[i])))

        import pickle
        if (flag):
            with open(outdir + '/edg.pkl', 'wb') as fp:
                pickle.dump(endict, fp, protocol=pickle.HIGHEST_PROTOCOL)
            with open(outdir + '/edge.pkl', 'wb') as fp:
                pickle.dump(edge_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)

            with open(outdir + '/nodef' + '.pkl', 'wb') as fp:
                pickle.dump(pinbasedfeats, fp, protocol=pickle.HIGHEST_PROTOCOL)

        np.save(outdir + '/condata_' + str(z) + '.npy', truecon)
        np.save(outdir + '/badcondata_' + str(z) + '.npy', fakecon)
        np.save(outdir + '/condata_mask_' + str(z) + '.npy', truecon_mask)
        np.save(outdir + '/badcondata_mask_' + str(z) + '.npy', fakecon_mask)
        np.save(outdir + '/condata_norm_' + str(z) + '.npy', truecon_norm)
        np.save(outdir + '/badcondata_norm_' + str(z) + '.npy', fakecon_norm)
        np.save(outdir + '/condata_mask_norm_' + str(z) + '.npy', truecon_mask_norm)
        np.save(outdir + '/badcondata_mask_norm_' + str(z) + '.npy', fakecon_mask_norm)
        if (flag):
            np.save(outdir + '/sizdata_x' + '.npy', trusize_x)
            np.save(outdir + '/sizdata_y' + '.npy', trusize_y)
            np.save(outdir + '/pdata' + '.npy', pincnt)

            flag = False
        np.save(outdir + '/xdata_' + str(z) + '.npy', truex)
        np.save(outdir + '/ydata_' + str(z) + '.npy', truey)
        print("ok")

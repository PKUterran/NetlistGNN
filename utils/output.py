import numpy as np
from scipy.stats import pearsonr, spearmanr, kendalltau

from typing import Dict, Any, Tuple, List
from sklearn.metrics import precision_score, recall_score, f1_score


def printout_xf1(arr1, arr2, prefix="", log_prefix="") -> Dict[str, Any]:
    pearsonr_rho, pearsonr_pval = pearsonr(arr1, arr2)
    spearmanr_rho, spearmanr_pval = spearmanr(arr1, arr2)
    kendalltau_rho, kendalltau_pval = kendalltau(arr1, arr2)
    mae = np.sum(np.abs(arr1 - arr2)) / len(arr1)
    delta = np.abs(arr1 - arr2)
    rmse = np.sqrt(np.sum(np.multiply(delta, delta)) / len(arr1))

    print(prefix + "pearson", pearsonr_rho, pearsonr_pval)
    print(prefix + "spearman", spearmanr_rho, spearmanr_pval)
    print(prefix + "kendall", kendalltau_rho, kendalltau_pval)
    print(prefix + "MAE", mae)
    print(prefix + "RMSE", rmse)
    return {
        f'{log_prefix}pearson_rho': pearsonr_rho,
        f'{log_prefix}pearsonr_pval': pearsonr_pval,
        f'{log_prefix}spearmanr_rho': spearmanr_rho,
        f'{log_prefix}spearmanr_pval': spearmanr_pval,
        f'{log_prefix}kendalltau_rho': kendalltau_rho,
        f'{log_prefix}kendalltau_pval': kendalltau_pval,
        f'{log_prefix}mae': mae,
        f'{log_prefix}rmse': rmse,
    }


def printout(arr1, arr2, prefix="", log_prefix="") -> Dict[str, float]:
    pearsonr_rho, pearsonr_pval = pearsonr(arr1, arr2)
    spearmanr_rho, spearmanr_pval = spearmanr(arr1, arr2)
    kendalltau_rho, kendalltau_pval = kendalltau(arr1, arr2)
    target = arr1 > 0.9
    source = arr2 > 0.9
#     print(np.mean(arr1), np.mean(arr2))
#     print(np.sum(target), np.sum(source), len(target))
    precision = precision_score(target, source)
    recall = recall_score(target, source)
    f1 = f1_score(target, source)
    mae = np.sum(np.abs(arr1 - arr2)) / len(arr1)
    delta = np.abs(arr1 - arr2)
    rmse = np.sqrt(np.sum(np.multiply(delta, delta)) / len(arr1))

    print(prefix + "pearson", pearsonr_rho, pearsonr_pval)
    print(prefix + "spearman", spearmanr_rho, spearmanr_pval)
    print(prefix + "kendall", kendalltau_rho, kendalltau_pval)
    print(prefix + "precision", precision)
    print(prefix + "recall", recall)
    print(prefix + "f1-score", f1)
    print(prefix + "MAE", mae)
    print(prefix + "RMSE", rmse)
    return {
        f'{log_prefix}pearson_rho': pearsonr_rho,
        f'{log_prefix}pearsonr_pval': pearsonr_pval,
        f'{log_prefix}spearmanr_rho': spearmanr_rho,
        f'{log_prefix}spearmanr_pval': spearmanr_pval,
        f'{log_prefix}kendalltau_rho': kendalltau_rho,
        f'{log_prefix}kendalltau_pval': kendalltau_pval,
        f'{log_prefix}precision': precision,
        f'{log_prefix}recall': recall,
        f'{log_prefix}f1': f1,
        f'{log_prefix}mae': mae,
        f'{log_prefix}rmse': rmse,
    }


def rademacher(intensity, numindices):
    arr = np.random.randint(low=0, high=2, size=numindices)
    return intensity * (2 * arr - 1)


def get_grid_level_corr(posandpred, binx, biny, xgridshape, ygridshape, set_name=''
                        ) -> Tuple[Dict[str, float], Dict[str, float]]:
    nodetarg, nodepred, posx, posy = [posandpred[:, i] for i in range(0, posandpred.shape[1])]
    cmap_tgt = np.zeros((xgridshape, ygridshape))
    cmap_prd, supp = np.zeros_like(cmap_tgt), np.zeros_like(cmap_tgt)
    wmap = 1e-6 * np.ones_like(cmap_tgt)
    indices = []
    for i in range(0, posandpred.shape[0]):
        key1, key2 = int(np.rint(posx[i] / binx)), int(np.rint(posy[i] / biny))
        if key1 == 0 and key2 == 0:
            continue
        wmap[key1][key2] += 1
        indices += [key2 + key1 * ygridshape]
        cmap_prd[key1][key2] += nodepred[i]
        cmap_tgt[key1][key2] += nodetarg[i]
    supp = np.clip(wmap, 0, 1)
    indices = list(set(indices))
    if 0 in indices:
        indices.remove(0)
    cmap_tgt = np.divide(cmap_tgt, wmap)
    cmap_prd = np.divide(cmap_prd, wmap)
    cmap_prd[0, 0] = 0
    cmap_tgt[0, 0] = 0
    wmap[0, 0] = 1e-6
    nctu, pred = cmap_tgt.flatten(), cmap_prd.flatten()
    getmask = np.zeros_like(nctu)
    getmask[indices] = 1
    nctu, pred = np.multiply(nctu, getmask), np.multiply(pred, getmask)
    #     printout(nctu[indices] + rademacher(1e-6, len(indices)), pred[indices] + rademacher(1e-6, len(indices)),
    #              "\t\tGRID_INDEX: ", f'{set_name}grid_index_')
    d1 = printout(nctu[indices], pred[indices],
                  "\t\tGRID_INDEX: ", f'{set_name}grid_index_')
    d2 = printout(nctu, pred, "\t\tGRID_NO_INDEX: ", f'{set_name}grid_no_index_')
    return d1, d2


def mean_dict(dicts: List[Dict[str, float]]) -> Dict[str, float]:
    full_dict: Dict[str, List[float]] = {}
    for d in dicts:
        for k, v in d.items():
            full_dict.setdefault(k, []).append(v)
    return {k: sum(vs) / len(vs) for k, vs in full_dict.items()}


if __name__ == '__main__':
    tgt = np.array([1.5, 1, 3, 2.2])
    src = np.array([1.6, 2.2, 1, 0.9])
    printout(tgt, src)

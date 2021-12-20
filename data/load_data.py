import numpy as np
import pickle
import torch
import dgl
import dgl.function as fn


def loadmanyarrays(listofnames, commonprefix):
    return [np.load(commonprefix + listofnames[i] + '.npy') for i in range(len(listofnames))]


def FOaverage(g):
    degrees = g.out_degrees(g.nodes()).type(torch.float32)
    g.ndata['addnlfeat'] = (g.ndata['feat']) / degrees.view(-1, 1)
    g.ndata['inter'] = torch.zeros_like(g.ndata['feat'])
    g.ndata['wts'] = torch.ones(g.number_of_nodes()) / degrees
    g.ndata['wtmsg'] = torch.zeros_like(g.ndata['wts'])
    g.update_all(message_func=fn.copy_src(src='addnlfeat', out='inter'),
                 reduce_func=fn.sum(msg='inter', out='addnlfeat'))
    g.update_all(message_func=fn.copy_src(src='wts', out='wtmsg'),
                 reduce_func=fn.sum(msg='wtmsg', out='wts'))
    hop1 = g.ndata['addnlfeat'] / (g.ndata['wts'].view(-1, 1))
    return hop1


def kthorderdegree(g, k):
    degreearr = np.zeros((g.number_of_nodes(), k))
    for z in range(0, g.number_of_nodes()):
        seeds = torch.LongTensor([z])
        for w in range(0, k):
            _, succs = g.out_edges(seeds)
            seeds = torch.cat([succs, seeds])
            degreearr[z, w] = torch.numel(torch.unique(seeds))
    return degreearr.astype(np.float32)


def hyperedge_from_nodelist(nodes):
    src, dst = [], []
    for i in range(0, len(nodes)):
        for j in range(0, len(nodes)):
            if j != i:
                src += [nodes[i], nodes[j]]
                dst += [nodes[j], nodes[i]]
    return src, dst


def prepare_data(dirname, giveniter, index, norming, outscalefac, logic_flag, hashcode, edgcap, degdim):
    posandfeats = loadmanyarrays(
        ['sizdata_x', 'sizdata_y', 'pdata', 'xdata_' + str(giveniter), 'ydata_' + str(giveniter)], dirname)
    norm_x, norm_y = posandfeats[-2] / np.max(posandfeats[-2]), posandfeats[-1] / np.max(posandfeats[-1])

    labels = np.load(dirname + 'iter_' + str(giveniter) + '_node_label_full_' + str(hashcode) + '_.npy')
    labels = labels[:, index].reshape(-1, 1)
    if norming:
        labels = outscalefac * labels / np.max(np.abs(labels))

    features = (
        np.hstack((posandfeats[0].reshape(-1, 1), posandfeats[1].reshape(-1, 1), posandfeats[2].reshape(-1, 1))).astype(
            np.float32)).reshape(-1, 3)

    if not logic_flag:
        features = (np.hstack((features, norm_x, norm_y)).astype(np.float32)).reshape(-1, 5)

    positions = (
        np.hstack((posandfeats[-2].reshape(-1, 1), posandfeats[-1].reshape(-1, 1))).astype(np.float32)).reshape(-1, 2)

    with open(dirname + "/edg.pkl", "rb") as input_file:
        edg = pickle.load(input_file)
    fullsrc, fulldst = [], []
    for e in edg.keys():
        if (len(edg[e]) > 1) and (len(edg[e]) < edgcap):
            u, v = hyperedge_from_nodelist(edg[e])
            fullsrc += u
            fulldst += v

    g = dgl.DGLGraph((torch.LongTensor(fullsrc), torch.LongTensor(fulldst)))
    g = dgl.to_homogeneous(dgl.transform.add_self_loop(g))

    g.ndata['feat'] = torch.from_numpy(features[:g.number_of_nodes()])
    g.ndata['rawpos'] = torch.from_numpy(positions[:g.number_of_nodes()])

    extra = FOaverage(g)
    print(np.shape(labels))
    g.ndata['label'] = torch.from_numpy(labels[:g.number_of_nodes()])

    idxarr = np.where(((posandfeats[-2])[:g.number_of_nodes()]) >= 2)[0]
    g.ndata['feat'] = torch.cat([g.ndata['feat'], extra], dim=1)
    if degdim >= 1:
        get = kthorderdegree(g, degdim)
        g.ndata['feat'] = torch.cat([g.ndata['feat'], torch.from_numpy(get)], dim=1)

    return g.subgraph(idxarr.astype(np.int32))

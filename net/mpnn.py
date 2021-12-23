import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import NNConv


from .naive import NaiveModel


class MPNNModel(nn.Module):
    def __init__(self, model_type, arch_detail, edge_dim, heads, activation, scalefac):
        super(MPNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.archlist = arch_detail
        self.heads = heads
        self.post = NaiveModel(self.archlist[-2] + self.archlist[0], activation, scalefac)
        for i in range(0, len(arch_detail) - 1):
            edge_func = nn.Linear(edge_dim, arch_detail[i] * arch_detail[i + 1])
            self.layers.append(NNConv(arch_detail[i], arch_detail[i + 1], edge_func))

    def wholeforward(self, g, x, xe):
        init_x = x
        for i in range(0, len(self.layers) - 1):
            x = F.tanh(self.layers[i](g, x, xe))
        feats = (torch.cat([x, init_x], dim=1))
        return self.post(feats)

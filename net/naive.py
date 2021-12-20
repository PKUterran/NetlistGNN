import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GATConv, SAGEConv, GraphConv


class NaiveModel(nn.Module):
    def __init__(self, feats, activation, scale):
        super(NaiveModel, self).__init__()
        self.first = nn.Linear(feats, 150)
        self.mid = nn.Linear(150, 150)
        self.midbn = nn.BatchNorm1d(100, eps=1e-2)
        self.lastbn = nn.BatchNorm1d(100, eps=1e-2)
        self.second = nn.Linear(150, 1)
        assert activation in ['sig', 'tanh']
        self.activation = activation
        self.scale = scale

    def forward(self, x):
        x = F.tanh((self.first(x)))
        x = F.tanh((self.mid(x)))
        if self.activation == 'sig':
            x = F.sigmoid(self.second(x))
        else:
            x = F.tanh(self.second(x))
        return self.scale * x


class TraditionalGNNModel(nn.Module):
    def __init__(self, model_type, arch_detail, heads, activation, scalefac):
        super(TraditionalGNNModel, self).__init__()
        self.layers = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.archlist = arch_detail
        self.heads = heads
        self.post = NaiveModel(self.archlist[-2] + self.archlist[0], activation, scalefac)
        for i in range(0, len(arch_detail) - 1):
            if model_type == 'SAGE':
                self.layers.append(SAGEConv(arch_detail[i], arch_detail[i + 1], aggregator_type='gcn'))
            elif model_type == 'GCN':
                self.layers.append(GraphConv(arch_detail[i], arch_detail[i + 1], allow_zero_in_degree=True))
                self.bns.append(nn.BatchNorm1d(arch_detail[i + 1]))
            else:
                if i == 0:
                    self.layers.append(
                        GATConv(arch_detail[i], arch_detail[i + 1], num_heads=self.heads, allow_zero_in_degree=True))
                elif i == len(arch_detail) - 2:
                    self.layers.append(GATConv(arch_detail[i] * self.heads, arch_detail[i + 1], num_heads=1,
                                               allow_zero_in_degree=True))
                else:
                    self.layers.append(GATConv(arch_detail[i] * self.heads, arch_detail[i + 1], num_heads=self.heads,
                                               allow_zero_in_degree=True))

    def wholeforward(self, g, x):
        init_x = x
        for i in range(0, len(self.layers) - 1):
            x = F.tanh(self.layers[i](g, x))
        feats = (torch.cat([x, init_x], dim=1))
        return self.post(feats)

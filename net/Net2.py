import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import GATConv
import dgl.function as fn


class MLP(nn.Module):
    def __init__(self, nfeats: int, hfeats: int, n_target: int, activation: str = 'sig'):
        super(MLP, self).__init__()
        self.lin1 = nn.Linear(nfeats, hfeats)
        self.lin2 = nn.Linear(hfeats, n_target)
        self.activation = activation

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        hfeat0 = g.ndata['hv']
        hfeat1 = self.lin1(hfeat0)
        output = self.lin2(F.relu(hfeat1))
        if self.activation == 'sig':
            output = F.sigmoid(output)
        elif self.activation == 'tanh':
            output = F.tanh(output)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output


class Net2f(nn.Module):
    def __init__(self, nfeats: int, hfeats: int, n_target: int, activation: str = 'sig'):
        super(Net2f, self).__init__()
        self.nfeat_lin = nn.Linear(nfeats, hfeats)
        self.gat1 = GATConv(hfeats, hfeats, 1)
        self.gat2 = GATConv(hfeats, hfeats, 1)
        self.gat3 = GATConv(hfeats, hfeats, 1)

        self.readout = nn.Linear(3 * hfeats, n_target)
        self.activation = activation

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        nfeat = g.ndata['hv']
        hnfeat0 = self.nfeat_lin(nfeat)
        hnfeat1 = self.gat1.forward(g, F.relu(hnfeat0))[:, 0, :]
        hnfeat2 = self.gat2.forward(g, F.relu(hnfeat1))[:, 0, :]
        hnfeat3 = self.gat3.forward(g, F.relu(hnfeat2))[:, 0, :]

        output = self.readout(torch.cat([
            hnfeat1, hnfeat2, hnfeat3
        ], dim=-1))

        if self.activation == 'sig':
            output = F.sigmoid(output)
        elif self.activation == 'tanh':
            output = F.tanh(output)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output


class Net2a(nn.Module):
    def __init__(self, nfeats: int, efeats: int, hfeats: int, n_target: int, activation: str = 'sig'):
        super(Net2a, self).__init__()
        self.nfeat_lin = nn.Linear(nfeats, hfeats)
        self.efeat_lin = nn.Linear(efeats, hfeats)
        self.gat1 = GATConv(hfeats, hfeats, 1)
        self.gat2 = GATConv(hfeats, hfeats, 1)
        self.gat3 = GATConv(hfeats, hfeats, 1)

        self.edge_lin1 = nn.Linear(hfeats, hfeats)
        self.edge_lin2 = nn.Linear(hfeats, hfeats)
        self.gat_e = GATConv(2 * hfeats, hfeats, 1)
        self.readout = nn.Linear(6 * hfeats, n_target)
        self.activation = activation

    def forward(self, g: dgl.DGLGraph) -> torch.Tensor:
        nfeat = g.ndata['hv']
        efeat = g.edata['he']
        hnfeat0 = self.nfeat_lin(nfeat)
        hefeat = self.efeat_lin(efeat)
        hnfeat1 = self.gat1.forward(g, F.relu(hnfeat0))[:, 0, :]
        hnfeat2 = self.gat2.forward(g, F.relu(hnfeat1))[:, 0, :]
        hnfeat3 = self.gat3.forward(g, F.relu(hnfeat2))[:, 0, :]

        g.ndata['hv_'] = hnfeat0
        g.ndata['he_'] = hefeat
        g.update_all(fn.copy_u('hv_', 'mv'), fn.mean('mv', 'v_mean'), fn.sum('mv', 'v_sum'))
        g.update_all(fn.copy_e('he_', 'me'), fn.mean('me', 'e_mean'), fn.sum('me', 'e_sum'))
        uev_mean = torch.cat([g.ndata['v_mean'], g.ndata['e_mean'], g.ndata['hv_']], dim=-1)
        uev_sum = torch.cat([g.ndata['v_sum'], g.ndata['e_sum'], g.ndata['hv_']], dim=-1)
        hnfeat_e_mean = self.edge_lin2(self.edge_lin1(uev_mean))
        hnfeat_e_sum = self.edge_lin2(self.edge_lin1(uev_sum))
        hnfeat_e1 = self.gat_e.forward(g, F.relu(torch.cat([hnfeat_e_mean, hnfeat_e_sum], dim=-1)))[:, 0, :]

        output = self.readout(torch.cat([
            hnfeat1, hnfeat2, hnfeat3,
            hnfeat_e_mean, hnfeat_e_sum, hnfeat_e1
        ], dim=-1))

        if self.activation == 'sig':
            output = F.sigmoid(output)
        elif self.activation == 'tanh':
            output = F.tanh(output)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output

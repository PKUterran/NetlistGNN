import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple


class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super(ResidualBlock, self).__init__()
        self.lin1 = nn.Linear(dim, dim)
        self.lin2 = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x2 = F.relu(self.lin2(F.relu(self.lin1(x))))
        return F.relu(x + x2)


class LatticeMPBlock(nn.Module):
    def __init__(self, dim):
        super(LatticeMPBlock, self).__init__()
        self.res = ResidualBlock(dim)
        self.lin = nn.Linear(dim, dim)
        self.lin_a = nn.Linear(dim, dim)

    def forward(self, v_c: torch.Tensor, na_cc: torch.Tensor) -> torch.Tensor:
        v_c = self.res.forward(v_c)
        v1_c = F.relu(self.lin(v_c))
        v2_c = F.relu(self.lin(na_cc @ v_c))
        return v1_c + v2_c


class FeatureGenBlock(nn.Module):
    def __init__(self, n_dim, c_dim, dim):
        super(FeatureGenBlock, self).__init__()
        self.lin1_n = nn.Linear(n_dim, dim)
        self.lin2_n = nn.Linear(dim, dim)
        self.lin1_c = nn.Linear(c_dim, dim)
        self.lin2_c = nn.Linear(2 * dim, dim)
        self.res_n = ResidualBlock(dim)
        self.res_c = ResidualBlock(dim)

    def forward(self, v_n: torch.Tensor, v_c: torch.Tensor, g_nc: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_n = self.res_n(F.relu(self.lin1_n(v_n)))
        v_c = self.res_n(F.relu(self.lin1_c(v_c)))
        v1_n = F.relu(self.lin2_n(v_n))
        v1_c = F.relu(self.lin2_c(torch.cat([v_c, g_nc @ v_n], dim=-1)))
        return v1_n, v1_c


class HyperMPBlock(nn.Module):
    def __init__(self, dim):
        super(HyperMPBlock, self).__init__()
        self.lin1_n = nn.Linear(dim, dim)
        self.lin2_n = nn.Linear(dim, dim)
        self.lin3_n = nn.Linear(2 * dim, dim)
        self.lin1_c = nn.Linear(dim, dim)
        self.lin2_c = nn.Linear(dim, dim)
        self.lin3_c = nn.Linear(2 * dim, dim)
        self.res1_n = ResidualBlock(dim)
        self.res2_n = ResidualBlock(dim)
        self.res1_c = ResidualBlock(dim)
        self.res2_c = ResidualBlock(dim)

    def forward(self, v_n: torch.Tensor, v_c: torch.Tensor, v1_n: torch.Tensor, v1_c: torch.Tensor,
                g_nc: torch.Tensor, g_cn: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        v_n = self.res1_n(v_n)
        v_c = self.res1_n(v_c)
        v_n = v_n + F.relu(self.lin3_n(torch.cat([
            F.relu(self.lin1_n(v1_n)),
            F.relu(self.lin2_n(g_cn @ v_c))
        ], dim=-1)))
        v_n = self.res2_n(v_n)
        v_c = self.res2_n(v_c)
        v_c = v_c + F.relu(self.lin3_c(torch.cat([
            F.relu(self.lin1_c(v1_c)),
            F.relu(self.lin2_c(g_nc @ v_n))
        ], dim=-1)))
        return v_n, v_c


class LHNN(nn.Module):
    def __init__(self, n_dim, c_dim, dim=32):
        super(LHNN, self).__init__()
        self.feature_gen = FeatureGenBlock(n_dim, c_dim, dim)
        self.hyper_mp_1 = HyperMPBlock(dim)
        self.hyper_mp_2 = HyperMPBlock(dim)
        self.lattice_mp = LatticeMPBlock(dim)
        self.lattice_mp_s1 = LatticeMPBlock(dim)
        self.lattice_mp_s2 = LatticeMPBlock(dim)
        self.v4_readout = nn.Linear(dim, 1)
        self.v6_readout = nn.Linear(dim, 1)

    def forward(self, v_n: torch.Tensor, v_c: torch.Tensor, g_nc: torch.Tensor, g_cn: torch.Tensor, na_cc: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        v1_n, v1_c = self.feature_gen.forward(v_n, v_c, g_nc)
        v2_n, v2_c = self.hyper_mp_1.forward(v1_n, v1_c, v1_n, v1_c, g_nc, g_cn)
        v3_n, v3_c = self.hyper_mp_2.forward(v2_n, v2_c, v1_n, v1_c, g_nc, g_cn)
        v4_c = self.lattice_mp.forward(v3_c, na_cc)
        v5_c = self.lattice_mp_s1.forward(v4_c, na_cc)
        v6_c = self.lattice_mp_s2.forward(v5_c, na_cc)
        v4_out = F.sigmoid(self.v4_readout(v4_c))
        v6_out = F.sigmoid(self.v4_readout(v6_c))
        return v6_out, v4_out

    @staticmethod
    def generate_adj(a_cc: torch.Tensor, h_nc: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # d_cc = torch.diag(torch.sum(h_nc, dim=1))
        b1_nn = torch.diag(torch.sum(h_nc, dim=0) ** -1)
        p1_cc = torch.diag(torch.sum(a_cc, dim=1) ** -1)
        g_nc = h_nc
        g_cn = b1_nn @ h_nc.t()
        na_cc = p1_cc @ a_cc
        return g_nc, g_cn, na_cc

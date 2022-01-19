import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv, SAGEConv, GATConv, HeteroGraphConv, GraphConv
from typing import Tuple, Dict, Any, List


class NodeNetGNN(nn.Module):
    def __init__(self, hidden_node_feats: int, hidden_net_feats: int, hidden_pin_feats: int,
                 out_node_feats: int, out_net_feats: int):
        super(NodeNetGNN, self).__init__()
        self.lin = nn.Linear(hidden_pin_feats, hidden_net_feats * out_node_feats)

        def edge_func(efeat):
            # print(efeat.shape)
            return self.lin(efeat)

        self.hetero_conv = HeteroGraphConv({
            'pins': GraphConv(in_feats=hidden_node_feats, out_feats=out_net_feats),
            'pinned': NNConv(in_feats=hidden_net_feats, out_feats=out_node_feats, edge_func=edge_func),
        })

    def forward(self, g: dgl.DGLHeteroGraph, node_feat: torch.Tensor, pin_feat: torch.Tensor, net_feat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h1 = {
            'node': node_feat,
            'net': net_feat,
        }
        h2 = self.hetero_conv.forward(g, h1, mod_kwargs={'pinned': {'efeat': pin_feat}})
        return h2['node'], h2['net']


class GridGNN(nn.Module):
    def __init__(self, in_feats: int, out_feats: int, num_heads: int):
        super(GridGNN, self).__init__()
        self.gat_conv = GATConv(
            in_feats=in_feats,
            out_feats=out_feats,
            num_heads=num_heads,
        )

    def forward(self, g: dgl.DGLGraph, feat: torch.Tensor) -> torch.Tensor:
        out_feat = self.gat_conv.forward(g, feat)
        return out_feat

    def forward_with_attention(self, g: dgl.DGLGraph, feat: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        out_feat, attn = self.gat_conv.forward(g, feat, get_attention=True)
        return out_feat, attn


class HyperGNN2D(nn.Module):
    def __init__(self, in_node_feats: int, in_net_feats: int, in_pin_feats: int, n_target: int, config: Dict[str, Any],
                 activation: str = 'sig'):
        super(HyperGNN2D, self).__init__()
        self.in_node_feats = in_node_feats
        self.in_net_feats = in_net_feats
        self.in_pin_feats = in_pin_feats
        self.n_layer = config['N_LAYER']
        self.out_node_feats = config['NODE_FEATS']
        self.out_net_feats = config['NET_FEATS']
        self.out_grid_feats = config['GRID_FEATS']
        self.grid_heads = config['GRID_HEADS']
        self.grid_channels = config['GRID_CHANNELS']
        self.hidden_node_feats = self.out_node_feats + self.out_grid_feats * self.grid_channels * self.grid_heads
        self.hidden_pin_feats = config['PIN_FEATS']
        self.hidden_net_feats = self.out_net_feats

        self.node_lin = nn.Linear(self.in_node_feats, self.hidden_node_feats)
        self.net_lin = nn.Linear(self.in_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.in_pin_feats, self.hidden_pin_feats)
        self.list_node_net_gnn = nn.ModuleList(
            [NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats, self.hidden_pin_feats,
                        self.out_node_feats, self.out_net_feats) for _ in range(self.n_layer)]
        )
        self.list_grid_gnn = nn.ModuleList(
            [GridGNN(self.hidden_node_feats, self.out_grid_feats, self.grid_heads) for _ in range(self.n_layer)]
        )
        self.n_target = n_target
        self.output_layer = nn.Linear(self.hidden_node_feats, self.n_target)
        self.activation = activation

    def forward(self, in_node_feat: torch.Tensor, in_net_feat: torch.Tensor, in_pin_feat: torch.Tensor,
                node_net_graph: dgl.DGLHeteroGraph = None,
                list_grid_graph: List[dgl.DGLGraph] = None,
                ) -> torch.Tensor:
        assert len(list_grid_graph) == self.grid_channels
        node_feat = F.leaky_relu(self.node_lin(in_node_feat))
        net_feat = F.leaky_relu(self.net_lin(in_net_feat))
        pin_feat = F.leaky_relu(self.pin_lin(in_pin_feat))

        for i in range(self.n_layer):
            out_node_feat, out_net_feat = self.list_node_net_gnn[i].forward(node_net_graph,
                                                                            node_feat, pin_feat, net_feat)
            list_out_grid_feat = [self.list_grid_gnn[i].forward(list_grid_graph[j], node_feat)
                                  for j in range(self.grid_channels)]
            out_grid_feat = torch.cat(list_out_grid_feat, dim=-1)
            out_grid_feat = torch.reshape(out_grid_feat,
                                          [-1, self.out_grid_feats * self.grid_channels * self.grid_heads])

            node_feat = torch.cat((out_node_feat, out_grid_feat), dim=-1)
            net_feat = out_net_feat

        output_predictions = self.output_layer(node_feat)
        if self.activation == 'sig':
            output_predictions = F.sigmoid(output_predictions)
        elif self.activation == 'tanh':
            output_predictions = F.tanh(output_predictions)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output_predictions

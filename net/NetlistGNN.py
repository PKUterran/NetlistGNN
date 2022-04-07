import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl.nn.pytorch import NNConv, SAGEConv, GATConv, HeteroGraphConv, GraphConv
from typing import Tuple, Dict, Any, List


class NodeNetGNN(nn.Module):
    def __init__(self, hidden_node_feats: int, hidden_net_feats: int, hidden_pin_feats: int,
                 out_node_feats: int, out_net_feats: int,
                 use_edge_attr=True):
        super(NodeNetGNN, self).__init__()
        self.use_edge_attr = use_edge_attr
        self.lin2 = nn.Linear(hidden_pin_feats, hidden_net_feats * out_node_feats)

        def edge_func2(efeat):
            return self.lin2(efeat)

        self.hetero_conv = HeteroGraphConv({
            'pins': GraphConv(in_feats=hidden_node_feats, out_feats=out_net_feats),
            'pinned': NNConv(in_feats=hidden_net_feats, out_feats=out_node_feats, edge_func=edge_func2)
            if use_edge_attr else GraphConv(in_feats=hidden_net_feats, out_feats=out_node_feats),
            'near': SAGEConv(in_feats=hidden_node_feats, out_feats=out_node_feats, aggregator_type='pool'),
        }, aggregate='max')

    def forward(self, g: dgl.DGLHeteroGraph, node_feat: torch.Tensor, pin_feat: torch.Tensor, net_feat: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        h = {
            'node': node_feat,
            'net': net_feat,
        }

        if self.use_edge_attr:
            h1 = self.hetero_conv.forward(g, h, mod_kwargs={
                'pinned': {'efeat': pin_feat},
            })
        else:
            h1 = self.hetero_conv.forward(g, h)

        return h1['node'], h1['net']


class NetlistGNN(nn.Module):
    def __init__(self, in_node_feats: int, in_net_feats: int, in_pin_feats: int, n_target: int, config: Dict[str, Any],
                 activation: str = 'sig', recurrent=False):
        super(NetlistGNN, self).__init__()
        self.recurrent = recurrent

        self.in_node_feats = in_node_feats
        self.in_net_feats = in_net_feats
        self.in_pin_feats = in_pin_feats
        self.n_layer = config['N_LAYER']
        self.out_node_feats = config['NODE_FEATS']
        self.out_net_feats = config['NET_FEATS']
        self.hidden_node_feats = self.out_node_feats
        self.hidden_pin_feats = config['PIN_FEATS']
        self.hidden_net_feats = self.out_net_feats

        self.node_lin = nn.Linear(self.in_node_feats, self.hidden_node_feats)
        self.net_lin = nn.Linear(self.in_net_feats, self.hidden_net_feats)
        self.pin_lin = nn.Linear(self.in_pin_feats, self.hidden_pin_feats)
        if self.recurrent:
            self.node_net_gnn = NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats, self.hidden_pin_feats,
                                           self.out_node_feats, self.out_net_feats)
        else:
            self.list_node_net_gnn = nn.ModuleList(
                [NodeNetGNN(self.hidden_node_feats, self.hidden_net_feats, self.hidden_pin_feats,
                            self.out_node_feats, self.out_net_feats) for _ in range(self.n_layer)])
        self.n_target = n_target
        self.output_layer_1 = nn.Linear(self.in_node_feats + self.hidden_node_feats, self.hidden_node_feats)
        self.output_layer_2 = nn.Linear(self.hidden_node_feats, self.hidden_node_feats)
        self.output_layer_3 = nn.Linear(self.hidden_node_feats, self.n_target)
        self.activation = activation

    def forward(self, in_node_feat: torch.Tensor, in_net_feat: torch.Tensor, in_pin_feat: torch.Tensor,
                node_net_graph: dgl.DGLHeteroGraph = None,
                ) -> torch.Tensor:
        node_feat = F.leaky_relu(self.node_lin(in_node_feat))
        net_feat = F.leaky_relu(self.net_lin(in_net_feat))
        pin_feat = F.leaky_relu(self.pin_lin(in_pin_feat))

        for i in range(self.n_layer):
            if self.recurrent:
                node_feat, net_feat = self.node_net_gnn.forward(node_net_graph, node_feat, pin_feat, net_feat)
            else:
                node_feat, net_feat = self.list_node_net_gnn[i].forward(node_net_graph, node_feat, pin_feat, net_feat)

        output_predictions = self.output_layer_3(torch.tanh(
            self.output_layer_2(torch.tanh(
                self.output_layer_1(torch.cat([in_node_feat, node_feat], dim=-1))
            ))
        ))
        if self.activation == 'sig':
            output_predictions = torch.sigmoid(output_predictions)
        elif self.activation == 'tanh':
            output_predictions = torch.tanh(output_predictions)
        else:
            assert False, f'Undefined activation {self.activation}'
        return output_predictions
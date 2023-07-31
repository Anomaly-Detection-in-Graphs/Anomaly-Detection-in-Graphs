from typing import Union

from torch import Tensor
from torch_sparse import SparseTensor
import torch
import torch.nn.functional as F
# from torch_geometric.nn import SAGEConv
from torch_geometric.nn import SAGEConv, GraphConv, ResGatedGraphConv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, DNAConv, SignedConv, GCN2Conv, GENConv, ClusterGCNConv, SuperGATConv, EGConv, GeneralConv, WLConvContinuous, FiLMConv
from torch_geometric.nn import GATConv, GATv2Conv, GCNConv

nonolist='AGNNConv GINConv GINEConv DNAConv SignedConv GCN2Conv EGConv'
cudalist = 'TransformerConv GENConv SuperGATConv GeneralConv'

class SAGE(torch.nn.Module):
    def __init__(self
                 , in_channels
                 , hidden_channels
                 , out_channels
                 , num_layers
                 , dropout
                 , batchnorm=True
                , aggr_type = 'max'):
        super(SAGE, self).__init__()
        SAGEConv = ResGatedGraphConv
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr = aggr_type))
        self.bns = torch.nn.ModuleList()
        self.batchnorm = batchnorm
        if self.batchnorm:
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr = aggr_type))
            if self.batchnorm:
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        # self.convs.append(torch.nn.Linear(hidden_channels, out_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels, aggr = aggr_type))

        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        if self.batchnorm:
            for bn in self.bns:
                bn.reset_parameters()

    def forward(self, x, edge_index: Union[Tensor, SparseTensor]):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            if self.batchnorm: 
                x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        # x = self.convs[-1](x)
        return x.log_softmax(dim=-1)
    
"""  
Arthur: @Zeyang Cui, Zihan Xie, Mingyang Zhao
""" 

import torch
import torch.nn.functional as F
import torch_geometric
import torch_scatter as scatter
from torch import Tensor
import numpy as np


def append_degree_info(x: Tensor, edge_index: Tensor):
    row, col = edge_index
    in_degree = torch_geometric.utils.degree(col, x.size(0), x.dtype)

    out_degree = torch_geometric.utils.degree(row, x.size(0), x.dtype)
    return torch.cat([x, in_degree.view(-1, 1), out_degree.view(-1, 1)], dim=1)


def append_label_counts(x, edge_index, y):
    y = y.clone().squeeze()
    background_nodes = torch.logical_or(y == 2, y == 3)
    foreground_nodes = torch.logical_and(y != 2, y != 3)
    y[background_nodes] = 1
    y[foreground_nodes] = 0

    row, col = edge_index
    a = F.one_hot(y[col])
    b = F.one_hot(y[row])
    temp = scatter.scatter(a, row, dim=0, dim_size=y.size(0), reduce="sum")
    temp += scatter.scatter(b, col, dim=0, dim_size=y.size(0), reduce="sum")

    return torch.cat([x, temp.to(x)], dim=1)


def cos_sim_sum(x, edge_index):
    row, col = edge_index
    sim = F.cosine_similarity(x[row], x[col])
    sim_sum = scatter.scatter(sim, row, dim=0, dim_size=x.size(0), reduce="sum")
    return torch.cat([x, torch.unsqueeze(sim_sum, dim=1)], dim=1)


def to_undirected(edge_index, edge_attr):

    row, col = edge_index
    row, col = torch.cat([row, col], dim=0), torch.cat([col, row], dim=0)
    edge_index = torch.stack([row, col], dim=0)

    edge_attr = torch.cat([edge_attr, edge_attr], dim=0)
    return edge_index, edge_attr


def data_process(data):
    edge_index, edge_attr = (
        data.edge_index,
        data.edge_attr
    )

    x = data.x
    x = append_degree_info(x, edge_index)
    x = cos_sim_sum(x, edge_index)
    edge_index, edge_attr = to_undirected(
        edge_index, edge_attr
    )
    mask = edge_index[0] < edge_index[1]
    edge_index = edge_index[:, mask]
    edge_attr = edge_attr[mask]
    data.edge_index, data.edge_attr = to_undirected(
        edge_index, edge_attr
    )

    data.edge_direct = torch.ones(data.edge_attr.size(0), dtype=torch.long)
    data.edge_direct[: data.edge_attr.size(0) // 2] = 0
    print(data.edge_attr.size())
    print(data.edge_index.size())

    x = append_label_counts(x, edge_index, data.y)
    
    # data.x = x
    data.x = x.float()
    # if data.y.dim() == 2:
    #     data.y = data.y.squeeze(1)
    return data

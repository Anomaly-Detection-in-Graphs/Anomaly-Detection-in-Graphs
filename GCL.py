""" 
Arthur: @Mingyang Zhao
""" 

import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv

from models.gcl import Encoder, Model, drop_feature
from eval import label_classification

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from logger import Logger


def train(model: Model, x, edge_index, is_DGraph = False):
    model.train()
    optimizer.zero_grad()
    edge_index_1 = dropout_adj(edge_index, p=drop_edge_rate_1)[0]
    edge_index_2 = dropout_adj(edge_index, p=drop_edge_rate_2)[0]
    x_1 = drop_feature(x, drop_feature_rate_1)
    x_2 = drop_feature(x, drop_feature_rate_2)
    z1 = model(x_1, edge_index_1)
    z2 = model(x_2, edge_index_2)

    loss = model.loss(z1, z2, batch_size=0)
    loss.backward()
    optimizer.step()

    return loss.item()


def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    print(label_classification(z, y, ratio=0.1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': F.relu, 'prelu': nn.PReLU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GATv2Conv': GATv2Conv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def get_dataset(path, name):
        assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
        name = 'dblp' if name == 'DBLP' else name

        return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())
    if args.dataset != "DGraphFin":
        path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
        dataset = get_dataset(path, args.dataset)
        data = dataset[0]
        print("dataset:", dataset,dataset[0], len(dataset))
    elif args.dataset == "DGraphFin":
        dataset = DGraphFin(root='~/DGraphFin_baseline-master/dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = dataset.num_classes
        if args.dataset in ['DGraphFin']: nlabels = 2
        print('dataset:', dataset)
        data = dataset[0]
        # print('data:', data)
        # print('data.edge_index', data.edge_index)
        data.edge_index = dataset.process().edge_index
        # data.edge_index = torch.load("../edge_index.pt")
        data.adj_t = data.adj_t.to_symmetric()
        
        if args.dataset in ['DGraphFin']:
            x = data.x
            x = (x-x.mean(0))/x.std(0)
            data.x = x
        if data.y.dim()==2:
            data.y = data.y.squeeze(1)        
        
        split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        train_idx = split_idx['train'].to(device)
        print('train_idx',train_idx, train_idx.size())

        data_edge_index = data.adj_t.to(device)
    dataset_num_features = dataset.num_features
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data = data.to(device)
    
    
    encoder = Encoder(dataset_num_features, num_hidden, activation,
                      base_model=base_model, k=num_layers).to(device)
    # print(dataset.num_features)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = t()
    prev = start
    if args.dataset == "DGraphFin":
        is_DGraph = True
    else:
        is_DGraph = False
#     for epoch in range(1, num_epochs + 1):
#         # loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, is_DGraph = is_DGraph)
#         loss = train(model, data.x, data.edge_index, is_DGraph = is_DGraph)

#         now = t()
#         print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
#               f'this epoch {now - prev:.4f}, total {now - start:.4f}')
#         prev = now

    print("=== Final ===")
    print(data.y)
    test(model, data.x.to(device), data.edge_index.to(device), data.y.to(device), final=True)

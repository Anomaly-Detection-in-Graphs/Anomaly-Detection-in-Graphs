"""  
Arthur: @Zeyang Cui, Zihan Xie, Mingyang Zhao
""" 


import argparse
import os.path as osp
import random
from time import perf_counter as t
import yaml
from yaml import SafeLoader
from tqdm import tqdm
import numpy as np
from utils.evaluator import Evaluator

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.data import NeighborSampler

from models.gcl_neighsampler import Encoder, Model, drop_feature
from eval import label_classification, mlp_label_classification

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from models.pGRACE.model_neighsampler import Encoder, GRACE, NewGConv, NewEncoder, NewGRACE
from models.pGRACE.functional import drop_feature, drop_edge_weighted, \
    degree_drop_weights, \
    evc_drop_weights, pr_drop_weights, \
    feature_drop_weights, drop_feature_weighted_2, feature_drop_weights_dense
from models.pGRACE.eval import log_regression, MulticlassEvaluator
from models.pGRACE.utils import get_base_model, get_activation, \
    generate_split, compute_pr, eigenvector_centrality
from models.pGRACE.dataset import get_dataset
from logger import Logger


def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False, is_DGraph = False, batch_size = 0):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
        
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        adjs_1 = [(dropout_adj(edge_index, p=drop_edge_rate_1)[0], batch, size) for edge_index, batch, size in adjs]
        adjs_2 = [(dropout_adj(edge_index, p=drop_edge_rate_2)[0], batch, size) for edge_index, batch, size in adjs]
        # if is_DGraph:
        #     x_1 = data.x[n_id]
        #     x_2 = data.x[n_id]
        # else:
        x_1 = drop_feature(data.x[n_id], drop_feature_rate_1)
        x_2 = drop_feature(data.x[n_id], drop_feature_rate_2)
        
        # print(batch_size, n_id)
        # print('adj_index',adjs[0][0].size())
        # print('_',adjs[0][1].size())
        # print('size',adjs[0][2])
        optimizer.zero_grad()
        z1 = model(x_1, adjs_1, [2, 2])
        z2 = model(x_2, adjs_2, [5, 5])
        loss = model.loss(z1, z2, batch_size=batch_size)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss



def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index, [1, 1], final=True)

    label_classification(z, y, ratio=0.1)
    
    
# @torch.no_grad()
def test_mini(layer_loader, model, data, y, evaluator, device, no_conv=False, is_DGraph = False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.encoder.inference(data.x, layer_loader, device)
    
    
    # losses, eval_results = dict(), dict()
    # for key in ['train', 'valid', 'test']:
    if is_DGraph:
        if use_mlp:
            split_idx = {'train':data.train_mask.to(device), 'valid':data.valid_mask.to(device), 'test':data.test_mask.to(device)}
            print(mlp_label_classification(out, y, split_idx = split_idx))
        else:
            node_id = torch.cat([data.train_mask, data.valid_mask, data.test_mask], dim=0)
            node_id = node_id.to(device)
            print(out[node_id].size())
            print('yyyyyyyyyyyyyy', y[node_id])
            out = out[node_id]
            y = y[node_id]
            print(label_classification(out, y, ratio=0.1))
    # sorted_y, _ = torch.sort(y[node_id], descending=True)
    # print(sorted_y)
    else:
        if use_mlp:
            if is_DGraph:
                split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
                print(mlp_label_classification(out, y, split_idx = split_idx))
            else:
                print(label_classification(out, y, ratio=0.1))
            
        else:
            print(label_classification(out, y, ratio=0.1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Cora')
    # parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--gpu_id', type=int, default=0)
    parser.add_argument('--config', type=str, default='config.yaml')
    args = parser.parse_args()

    assert args.gpu_id in range(0, 8)
    torch.cuda.set_device(args.gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = yaml.load(open(args.config), Loader=SafeLoader)[args.dataset]

    torch.manual_seed(config['seed'])
    random.seed(12345)

    learning_rate = config['learning_rate']
    num_hidden = config['num_hidden']
    num_proj_hidden = config['num_proj_hidden']
    activation = ({'relu': 'relu', 'prelu': 'prelu', 'rrelu': 'rrelu'})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GATv2Conv': GATv2Conv})[config['base_model']]
    num_layers = config['num_layers']

    drop_edge_rate_1 = config['drop_edge_rate_1']
    drop_edge_rate_2 = config['drop_edge_rate_2']
    drop_feature_rate_1 = config['drop_feature_rate_1']
    drop_feature_rate_2 = config['drop_feature_rate_2']
    tau = config['tau']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']

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
        train_idx = np.array(list(range(len(data.x))))
        train_idx = torch.LongTensor(train_idx).to(device)
        # print("dataset:", dataset,dataset[0], len(dataset))
    elif args.dataset == "DGraphFin":
        dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = dataset.num_classes
        if args.dataset in ['DGraphFin']: nlabels = 2
        # print('dataset:', dataset)
        data = dataset[0]
        # print('data:', data)
        # print('data.edge_index', data.edge_index)
        data.edge_index = dataset.process().edge_index
        # data.edge_index = torch.load("../edge_index.pt")
        data.adj_t = data.adj_t.to_symmetric()
        data_edge_index = data.adj_t
        
        if args.dataset in ['DGraphFin']:
            x = data.x
            x = (x-x.mean(0))/x.std(0)
            data.x = x
        if data.y.dim()==2:
            data.y = data.y.squeeze(1)        
        
        split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        train_idx = np.array(list(range(3700550)))
        # print('maskkkkkkkkkkkkkkkkk', len(data.valid_mask))
        # print(data.valid_mask)
        # train_idx = train_idx[~np.in1d(train_idx, data.train_mask.cpu().detach().numpy())]
        # train_idx = train_idx[~np.in1d(train_idx, data.valid_mask.cpu().detach().numpy())]
        # train_idx = train_idx[~np.in1d(train_idx, data.test_mask.cpu().detach().numpy())]
        train_idx = torch.LongTensor(train_idx).to(device)
    
    data = data.to(device)
    print(len(data.x))
    print(train_idx)
    # train_idx = split_idx['train'].to(device)
    batch_size = 8192
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[2,10], batch_size=batch_size, shuffle=True, num_workers=12)
    layer_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12) 
    
    # print(data_edge_index.size,data_x.size())
    dataset_num_features = dataset.num_features
    # print('numnumnumnum', dataset_num_features)
    encoder = NewEncoder(dataset.num_features, num_hidden, get_activation(activation),
                      base_model=NewGConv, k=num_layers).to(device)
    adj = 0

    model = NewGRACE(encoder, adj, num_hidden, num_proj_hidden, tau).to(device)
    
    # model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = t()
    prev = start
    if args.dataset == "DGraphFin":
        is_DGraph = True
    else:
        is_DGraph = False
    for epoch in range(1, num_epochs + 1):
        loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False, batch_size = batch_size, is_DGraph = is_DGraph)
        
        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now

    print("=== Final ===")

#     test(model, data.x, data.edge_index, data.y, final=True)
    test_mini(layer_loader, model, data, data.y, Evaluator, device, no_conv=False, is_DGraph = is_DGraph, use_mlp = True)

"""
Arthur: @Mingyang Zhao
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
from torch_geometric.utils import dropout_adj, shuffle_node, dropout_node, dropout_edge
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.nn import GraphConv, ResGatedGraphConv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, DNAConv, SignedConv, GCN2Conv, GENConv, ClusterGCNConv, SuperGATConv, EGConv, GeneralConv
# from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborLoader, NeighborSampler, LinkNeighborLoader, ImbalancedSampler, ClusterData, ClusterLoader, GraphSAINTSampler, GraphSAINTRandomWalkSampler, GraphSAINTNodeSampler, ShaDowKHopSampler, RandomNodeLoader
from torch_geometric.data import Batch

from models.gcl_neighsampler import Encoder, Model, drop_feature
from eval import label_classification, mlp_label_classification

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from logger import Logger
from gnn import train as gnn_train
from gnn import test as gnn_test


mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

gcn_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

sage_parameters = {'lr':0.01
              , 'num_layers':1
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }

def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False, is_DGraph = False, batch_size = 0):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
#     for sampled_hetero_data in train_loader:
#         batch_size = sampled_hetero_data.batch_size
        
#         x_1 = drop_feature(sampled_hetero_data.x, drop_feature_rate_1)
#         x_2, _ = shuffle_node(sampled_hetero_data.x)
        
        # print(sampled_hetero_data)
        # print('n_iddddddd',n_id)
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        adjs = [adj.to(device) for adj in adjs]
        # for edge_index, batch, size in adjs:
        #     print('edge size', size)
        #     print('size', data.x[n_id].size())
        #     print('edge size', edge_index.size())
        adjs_1 = [(dropout_adj(edge_index, p=drop_edge_rate_1)[0], batch, size) for edge_index, batch, size in adjs]
        adjs_2 = [(dropout_adj(edge_index, p=drop_edge_rate_2)[0], batch, size) for edge_index, batch, size in adjs]
        
        x_1 = drop_feature(data.x[n_id], drop_feature_rate_1)
        x_2 = drop_feature(data.x[n_id], drop_feature_rate_2)
        # x_2, _ = shuffle_node(data.x[n_id])
     
#         x_1 = []
#         x_2 = []
#         adjs_1 = []
#         adjs_2 = []
        
#         for edge_index, batch, size in adjs:
#             # print('size', data.x[n_id].size())
#             # print('edge size', edge_index.size())
#             edge_index, edge_mask, node_mask = dropout_node(edge_index, p=drop_node_rate_1)
#             # masked_indices = np.where(node_mask)[0]
#             # masked_nodes = n_id[~np.isin(n_id, masked_indices)]
#             x_1.append(data.x[n_id])
#             adjs_1.append((edge_index, batch, size))
# #             print('size', data.x[masked_nodes].size())
# #             print('edge size', edge_index.size())
            
#             edge_index, edge_mask, node_mask = dropout_node(edge_index, p=drop_node_rate_2)
#             # masked_indices = np.where(node_mask)[0]
#             # masked_nodes = n_id[~np.isin(n_id, masked_indices)]
#             x_2.append(data.x[n_id])
#             adjs_2.append((edge_index, batch, size))
            
        # x_1 = [(dropout_node(edge_index, p=drop_node_rate_1)[0], batch, size) for edge_index, batch, size in adjs]
        # x_2 = [(dropout_node(edge_index, p=drop_node_rate_2)[0], batch, size) for edge_index, batch, size in adjs]
        
        
        
        
        # print(batch_size, n_id)
        # print('adj_index',adjs[0][0].size())
        # print('_',adjs[0][1].size())
        # print('size',adjs[0][2])
        optimizer.zero_grad()
        z1 = model(x_1, adjs_1)[:batch_size*100]
        z2 = model(x_2, adjs_2)[:batch_size*100]
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
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)
    
    
# @torch.no_grad()
def test_mini(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    model.eval()
    
    out = model.encoder.inference(data.x, layer_loader, device)
    
    
    # losses, eval_results = dict(), dict()
    # for key in ['train', 'valid', 'test']:
    if args.dataset == "DGraphFin":
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
            split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
            print(mlp_label_classification(out, y, split_idx = split_idx))
        else:
            print(label_classification(out, y, ratio=0.1))
            
def test_mini4training(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
#     # data.y is labels of shape (N, ) 
#     model.eval()
    
#     data.x = model.encoder.inference(data.x, layer_loader, device)
#     data.x = data.x.detach().cpu().numpy()
#     data.x = torch.tensor(data.x).to(device)
    
    model.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    with torch.no_grad():
        # print('xxxxxxxxxx',data.x)

        data.x = model.encoder.inference(data.x, layer_loader, device)
        data.x = data.x.detach().cpu().numpy()
        data.x = torch.tensor(data.x).to(device)
    
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')   
    detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
    optimizer = torch.optim.Adam(detect_model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    loss_function = torch.nn.CrossEntropyLoss()
    best_valid = 0
    min_valid_loss = 1e8
    best_out = None
    
    split_idx = {'train':data.train_mask.cpu(), 'valid':data.valid_mask.cpu(), 'test':data.test_mask.cpu()}
    train_idx = split_idx['train'].to(device)
    evaluator = evaluator('auc')
    
    for epoch in range(1, 700+1):
        # data.y = torch.randint(0,2, (data.y.size(0),)).to(device)
        # data.x = torch.rand(data.x.size(0),data.x.size(1)).to(device)
        detect_model.train()

        optimizer.zero_grad()
        if no_conv:

            print('data_x.size',data.x[train_idx].size())
            out = detect_model(data.x[train_idx])
        else:
            print('data_adj.size',data.adj_t.size)
            print('data_x.size',data.x.size())
            out = detect_model(data.x, data.adj_t)[train_idx]
        loss = loss_function(out, data.y[train_idx])
        # loss = F.nll_loss(out, data.y[train_idx])
        loss.backward()
        optimizer.step()
    
        total_loss = loss.item()
        eval_results, losses, out = gnn_test(detect_model, data, split_idx, evaluator, no_conv)
        train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_out = out.cpu()
            train_best_eval, valid_best_eval, test_best_eval = train_eval, valid_eval, test_eval
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {total_loss:.4f}, '
                f'Train: {100 * train_eval:.3f}%, '
                f'Valid: {100 * valid_eval:.3f}% '
                f'Test: {100 * test_eval:.3f}%')
        print(f'Best_Train: {100 * train_best_eval:.3f}%, '
                f'Best_Valid: {100 * valid_best_eval:.3f}% '
                f'Best_Test: {100 * test_best_eval:.3f}%')
        

        
def test_loader_mini4training(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    
    model.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    with torch.no_grad():
        # print('xxxxxxxxxx',data.x)

        data.x = model.encoder.inference(data.x, layer_loader, device)
        data.x = data.x.detach().cpu().numpy()
        data.x = torch.tensor(data.x).to(device)
    # print('xxxxxxxxxx',data.x)
    
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')   
    detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
    optimizer = torch.optim.Adam(detect_model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    loss_function = torch.nn.CrossEntropyLoss()
    best_valid = 0
    min_valid_loss = 1e8
    best_out = None
    
    split_idx = {'train':data.train_mask.cpu(), 'valid':data.valid_mask.cpu(), 'test':data.test_mask.cpu()}
    train_idx = split_idx['train'].to(device)
    evaluator = evaluator('auc')
    # print('xxxxxxxxxx222222',data.x)
    
    for epoch in range(1, 1200+1):
        # data.y = torch.randint(0,2, (data.y.size(0),)).to(device)
        # data.x = torch.rand(data.x.size(0),data.x.size(1)).to(device)
        detect_model.train()

        optimizer.zero_grad()
        # print('xxxxxxxxxx33333333333',data.x)
        if no_conv:

            print('data_x.size',data.x[train_idx].size())
            out = detect_model(data.x[train_idx])
        else:
            print('data_x.size',data.x.size())
            out = detect_model(data.x, data.edge_index)[train_idx]

        loss = loss_function(out, data.y[train_idx])
        # loss = F.nll_loss(out, data.y[train_idx])
        loss.backward()
        optimizer.step()
    
        total_loss = loss.item()
        eval_results, losses, out = gnn_test(detect_model, data, split_idx, evaluator, no_conv, is_loader = True)
        train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_out = out.cpu()
            train_best_eval, valid_best_eval, test_best_eval = train_eval, valid_eval, test_eval
        if epoch % 1 == 0:
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {total_loss:.4f}, '
                f'Train: {100 * train_eval:.3f}%, '
                f'Valid: {100 * valid_eval:.3f}% '
                f'Test: {100 * test_eval:.3f}%')
        print(f'Best_Train: {100 * train_best_eval:.3f}%, '
                f'Best_Valid: {100 * valid_best_eval:.3f}% '
                f'Best_Test: {100 * test_best_eval:.3f}%')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='Cora')
    parser.add_argument('--dataset', type=str, default='DGraphFin')
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
    activation = ({'relu': F.relu, 'prelu': nn.PReLU(), 'elu':nn.ELU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GATv2Conv': GATv2Conv, 'TAGConv': TAGConv, 'ResGatedGraphConv': ResGatedGraphConv, 'GraphConv': GraphConv, 'GeneralConv': GeneralConv, 'GENConv': GENConv})[config['base_model']]
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
        nlabels = dataset.num_classes
        args.nlabels = nlabels
        
        data = dataset[0]
        train_idx = np.array(list(range(len(data.x))))
        train_idx = torch.LongTensor(train_idx).to(device)
        # print("dataset:", dataset,dataset[0], len(dataset))
    elif args.dataset == "DGraphFin":
        drop_node_rate_1 = config['drop_node_rate_1']
        drop_node_rate_2 = config['drop_node_rate_2']
        dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = 2
        args.nlabels = nlabels
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
        # train_idx = np.array(list(range(3700550)))
        # train_idx = train_idx[~np.in1d(train_idx, data.train_mask.cpu().detach().numpy())]
        # train_idx = train_idx[~np.in1d(train_idx, data.valid_mask.cpu().detach().numpy())]
        # train_idx = train_idx[~np.in1d(train_idx, data.test_mask.cpu().detach().numpy())]
        # train_idx = torch.LongTensor(train_idx).to(device)
        train_idx = torch.cat((data.train_mask, data.valid_mask, data.test_mask), dim=0)
        # .to(device)
        print('lengthhhhh',len(train_idx))
    
    data = data.to(device)
    
    # train_idx = split_idx['train'].to(device)
    batch_size = 1024
    
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[3,10], batch_size=batch_size, shuffle=True, num_workers=24)
    layer_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=2048, shuffle=False, num_workers=24) 

    
    # print(data_edge_index.size,data_x.size())
    dataset_num_features = dataset.num_features
    # print('numnumnumnum', dataset_num_features)
    encoder = Encoder(dataset_num_features, num_hidden, activation,base_model=base_model, k=num_layers).to(device)
    # print(dataset.num_features)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    model.reset_parameters()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    start = t()
    prev = start
    if args.dataset == "DGraphFin":
        is_DGraph = True
    else:
        is_DGraph = False
    mini_loss = 1*10000
#     for epoch in range(1, num_epochs + 1):
#         loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False, batch_size = batch_size, is_DGraph = is_DGraph)

#         now = t()
#         print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
#               f'this epoch {now - prev:.4f}, total {now - start:.4f}')
#         prev = now
#         if loss < mini_loss:
#             mini_loss = loss
# #             torch.save(model, "best_embedding.pt")

    print("=== Final ===")

#     test(model, data.x, data.edge_index, data.y, final=True)
    # test_mini(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    test_mini4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)

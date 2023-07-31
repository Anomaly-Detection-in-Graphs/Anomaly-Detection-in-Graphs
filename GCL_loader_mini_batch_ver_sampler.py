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
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report, roc_auc_score
from accelerate import Accelerator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize, OneHotEncoder

from models.gcl import Encoder, Model, drop_feature
from eval import label_classification, mlp_label_classification

from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from logger import Logger
from gnn import train as gnn_train
from gnn import test as gnn_test
from torch_geometric.nn import GraphConv, ResGatedGraphConv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, DNAConv, SignedConv, GCN2Conv, GENConv, ClusterGCNConv, SuperGATConv, EGConv, GeneralConv
from feature_processing import data_process
import os
import time

os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 


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
              , 'hidden_channels':96
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
              # , 'aggr': 'max'
             }

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def train1(epoch, train_loader,train_loader1, model, data, train_idx, optimizer, device, no_conv=False, is_DGraph = False, batch_size = 0):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for sampled_data,sampled_data1 in zip(train_loader,train_loader1):
        sampled_data = sampled_data.to(device)
        sampled_data1 = sampled_data1.to(device)
        
        adjs_1, edge_mask_1 = dropout_edge(sampled_data.edge_index, p=drop_edge_rate_1)
        adjs_1 = adjs_1.to(device)
        adjs_2, edge_mask_2 = dropout_edge(sampled_data.edge_index, p=drop_edge_rate_2)
        adjs_2 = adjs_2.to(device)

        
        x_1 = drop_feature(sampled_data.x, drop_feature_rate_1)
        x_2 = drop_feature(sampled_data.x, drop_feature_rate_2)
        
        adjs_11, edge_mask_11 = dropout_edge(sampled_data1.edge_index, p=drop_edge_rate_1)
        adjs_1 = adjs_1.to(device)
        adjs_21, edge_mask_21 = dropout_edge(sampled_data1.edge_index, p=drop_edge_rate_2)
        adjs_21 = adjs_21.to(device)
        
        x_11 = drop_feature(sampled_data1.x, drop_feature_rate_1)
        x_21 = drop_feature(sampled_data1.x, drop_feature_rate_2)
        
        
        optimizer.zero_grad()
        z1 = model(x_1, adjs_1)[:batch_size]
        z2 = model(x_2, adjs_2)[:batch_size]
        
        z11 = model(x_11, adjs_11)[:batch_size]
        z21 = model(x_21, adjs_21)[:batch_size]
        loss = model.loss(z1, z2)  + model.loss(z11, z21)
        loss.backward()
        optimizer.step()

        total_loss += float(loss)
        pbar.update(batch_size)

    pbar.close()
    loss = total_loss / len(train_loader)

    return loss

def train(epoch, train_loader, model, data, train_idx, optimizer, device, no_conv=False, is_DGraph = False, batch_size = 0):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for sampled_data in train_loader:
        sampled_data = sampled_data.to(device)
        
#         x_1 = drop_feature(sampled_hetero_data.x, drop_feature_rate_1)
#         x_2, _ = shuffle_node(sampled_hetero_data.x)
        
        # print(sampled_hetero_data)
        # print('n_iddddddd',n_id)
        # `adjs` holds a list of `(edge_index, e_id, size)` tuples.
        # adjs = [adj.to(device) for adj in adjs]
        # for edge_index, batch, size in adjs:
        #     print('edge size', size)
        #     print('size', data.x[n_id].size())
        #     print('edge size', edge_index.size())
        adjs_1, edge_mask_1 = dropout_edge(sampled_data.edge_index, p=drop_edge_rate_1)
        # adjs_1, edge_mask_1 = adjs_1.to(device), edge_mask_1.to(device)
        adjs_1 = adjs_1.to(device)
        adjs_2, edge_mask_2 = dropout_edge(sampled_data.edge_index, p=drop_edge_rate_2)
        # adjs_2, edge_mask_2 = adjs_2.to(device), edge_mask_2.to(device)
        adjs_2 = adjs_2.to(device)
        # edge_attr1 = sampled_data.edge_attr[edge_mask_1]
        # edge_attr2 = sampled_data.edge_attr[edge_mask_2]
        
        x_1 = drop_feature(sampled_data.x, drop_feature_rate_1)
        x_2 = drop_feature(sampled_data.x, drop_feature_rate_2)
        # x_2, _ = shuffle_node(sampled_data.x)
     
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
        
        
        
        
        optimizer.zero_grad()
        # print('shapeeee',torch.cat((sampled_data.train_mask, sampled_data.valid_mask, sampled_data.test_mask), dim=0).size())
        z1 = model(x_1, adjs_1)[:batch_size*12]
        # [:batch_size*128]
        z2 = model(x_2, adjs_2)[:batch_size*12]
        # [:batch_size*128]
        
        # [:sampled_data.batch_size]
        # z1 = model(x_1, adjs_1, edge_attr1)[:sampled_data.batch_size]
        # z2 = model(x_2, adjs_2, edge_attr2)[:sampled_data.batch_size]
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

    print(label_classification(z, y, ratio=0.1))
    
    
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
            

def test_mini_loader_4training(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
    eval_method = 'auc'
    # device = 'cpu'
    data = data.to('cpu')
    
    print(data.x.size())

    
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')   
    detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    # detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
    optimizer = torch.optim.Adam(detect_model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    loss_function = torch.nn.CrossEntropyLoss()
    n_samples = data.x.shape[0]
    
    if args.dataset == "DGraphFin":
        split_idx = {'train':data.train_mask.long(), 'valid':data.valid_mask.long(), 'test':data.test_mask.long()}
        train_idx = split_idx['train']
        # print(torch.max(data.y[train_idx]))
        batch_size= 8196
        print(data.__dict__)
        
        
        train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[10,10], batch_size=batch_size, num_workers=10, shuffle=True)
        eval_train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
        eval_loader = NeighborLoader(data, input_nodes=split_idx['valid'], num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
        test_loader = NeighborLoader(data, input_nodes=split_idx['test'], num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
        
       
    else:
        print(data.__dict__)
        ratio = 0.1
        valid_size = int(n_samples*ratio)
        test_size = int(n_samples*ratio)
        train_size = n_samples - valid_size - test_size
        indices = np.arange(n_samples)
        data_train, data_test, labels_train, labels_test, indices_train, indices_test = train_test_split(data.x, data.y, indices, test_size=test_size)
        data_train, data_val, labels_train, labels_val, indices_train, indices_val = train_test_split(data_train, labels_train, indices_train, test_size=valid_size)
        split_idx = {'train':torch.from_numpy(indices_train), 'valid':torch.from_numpy(indices_val), 'test':torch.from_numpy(indices_test)}
    
        train_idx = split_idx['train']
        print(torch.max(data.y[train_idx]))
        batch_size= 8196
        train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[10,10], batch_size=batch_size, num_workers=10, shuffle=True)
        eval_train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
        eval_loader = NeighborLoader(data, input_nodes=split_idx['valid'], num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
        test_loader = NeighborLoader(data, input_nodes=split_idx['test'], num_neighbors=[-1, -1], batch_size=batch_size, num_workers=10, shuffle=False)
    
    best_valid = 0
    min_valid_loss = 1e12
    
    
    evaluator = evaluator(eval_method)
    counter = 0
    for epoch in range(1, 700+1):
        detect_model.train()
        pred = []
        total_loss = 0
        
        start_time = time.perf_counter()
        for batched_sampled_data in train_loader:
            # print('hello')
            # print(batched_sampled_data.y.size(), batched_sampled_data.x.size())
            # print(batched_sampled_data.y[:batch_size])
            batched_sampled_data = batched_sampled_data.to(device)
            
            optimizer.zero_grad()
            # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
            if no_conv:
                # print('data_x.size',sampled_data.x.size())
                out = detect_model(batched_sampled_data.x)
            else:
                # out = detect_model(sampled_data.x, sampled_data.edge_index)
                out = detect_model(batched_sampled_data.x, batched_sampled_data.edge_index)
                
            
            
            loss = loss_function(out[:batched_sampled_data.batch_size], batched_sampled_data.y[:batched_sampled_data.batch_size])
            # loss = F.nll_loss(out, data.y[train_idx])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
             
         
        print('evalllllllllllllllllllllllll', counter)
        counter += 1
        valid_loss = 0
        detect_model.eval()
        for batched_sampled_data in eval_loader:
            batched_sampled_data = batched_sampled_data.to(device)
            # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
            if no_conv:
                # print('data_x.size',sampled_data.x.size())
                out = detect_model(batched_sampled_data.x)
            else:
                # out = detect_model(sampled_data.x, sampled_data.edge_index)
                out = detect_model(batched_sampled_data.x, batched_sampled_data.edge_index)
                
            valid_loss += loss_function(out[:batched_sampled_data.batch_size], batched_sampled_data.y[:batched_sampled_data.batch_size]).item()
        if counter > 40:
            break
            

        if valid_loss < min_valid_loss:
            counter = 0
            min_valid_loss = valid_loss
            eval_results, losses = gnn_test_loader(detect_model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, args, no_conv, is_loader=False)
            train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']
            train_best_eval, valid_best_eval, test_best_eval = train_eval, valid_eval, test_eval
            print(f'Epoch: {epoch:02d}, '
                f'Loss: {total_loss:.4f}, '
                f'Train: {100 * train_eval:.3f}%, '
                f'Valid: {100 * valid_eval:.3f}% '
                f'Test: {100 * test_eval:.3f}%')
        print(f'Best_Train: {100 * train_best_eval:.3f}%, '
                f'Best_Valid: {100 * valid_best_eval:.3f}% '
                f'Best_Test: {100 * test_best_eval:.3f}%')
        end_time = time.perf_counter()
        print('duration', end_time - start_time)
        
@torch.no_grad()
def gnn_test_loader(model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, args, no_conv=False, is_loader=False):
    # data.y is labels of shape (N, )
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    losses, eval_results = dict(), dict()
    train_pred, eval_pred, test_pred = [], [], []
    losses['train'], losses['valid'], losses['test'] = 0, 0, 0
    eval_method = 'auc'
    
    node_id = split_idx['train']
    for batched_sampled_data in eval_train_loader:
        batched_sampled_data = batched_sampled_data.to(device)
        # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
        if no_conv:
            # print('data_x.size',sampled_data.x.size())
            out = model(batched_sampled_data.x)[:batched_sampled_data.batch_size]
        else:
            # out = detect_model(sampled_data.x, sampled_data.edge_index)
            out = model(batched_sampled_data.x, batched_sampled_data.edge_index)[:batched_sampled_data.batch_size]
                
        losses['train'] += loss_function(out, batched_sampled_data.y[:batched_sampled_data.batch_size].to(device)).item()
        train_pred.append(out.detach().cpu().exp())
    y_train_pred = torch.cat(train_pred, dim=0)
    eval_results['train'] = evaluator.eval(data.y[node_id], y_train_pred)[eval_method]
    
    
    node_id = split_idx['valid']
    for batched_sampled_data in eval_loader:
        batched_sampled_data = batched_sampled_data.to(device)
        # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
        if no_conv:
            # print('data_x.size',sampled_data.x.size())
            out = model(batched_sampled_data.x)[:batched_sampled_data.batch_size]
        else:
            # out = detect_model(sampled_data.x, sampled_data.edge_index)
            out = model(batched_sampled_data.x, batched_sampled_data.edge_index)[:batched_sampled_data.batch_size]
            
        losses['valid'] += loss_function(out, batched_sampled_data.y[:batched_sampled_data.batch_size].to(device)).item()
        # out = out.exp()
        eval_pred.append(out.detach().cpu().exp())
    y_eval_pred = torch.cat(eval_pred, dim=0)
    eval_results['valid'] = evaluator.eval(data.y[node_id], y_eval_pred)[eval_method]
    
    node_id = split_idx['test']
    for batched_sampled_data in test_loader:
        batched_sampled_data = batched_sampled_data.to(device)
        # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
        if no_conv:
            # print('data_x.size',sampled_data.x.size())
            out = model(batched_sampled_data.x)[:batched_sampled_data.batch_size]
        else:
            # out = detect_model(sampled_data.x, sampled_data.edge_index)
            out = model(batched_sampled_data.x, batched_sampled_data.edge_index)[:batched_sampled_data.batch_size]
            
        losses['test'] += loss_function(out, batched_sampled_data.y[:batched_sampled_data.batch_size].to(device)).item()
        test_pred.append(out.detach().cpu().exp())
    y_test_pred = torch.cat(test_pred, dim=0)
    eval_results['test'] = evaluator.eval(data.y[node_id], y_test_pred)[eval_method]
    
    y_discrete_pred = y_test_pred.argmax(axis=-1)

    sorted_y = np.array(y_discrete_pred)
    print('outoutout', np.sort(sorted_y))
    y_pred_auc = y_test_pred.detach().cpu().numpy()
    y_pred_auc = y_pred_auc[:, 1]
    y_test = data.y[node_id].detach().cpu().numpy()
    y_test = y_test.reshape(-1, 1)
    
    micro = f1_score(y_test, y_discrete_pred, average="micro")
    macro = f1_score(y_test, y_discrete_pred, average="macro")
    recall = recall_score(y_test, y_discrete_pred, labels = [0,1], average=None)
    precision = precision_score(y_test, y_discrete_pred, labels = [0,1], average=None)
    cm = classification_report(y_test, y_discrete_pred)
    
    if args.dataset == "DGraphFin":
        auc = roc_auc_score(y_test, y_pred_auc, average = None)
        auc_weighted = roc_auc_score(y_test, y_pred_auc, average = 'weighted')
    else:
        y_pred_auc = prob_to_one_hot(y_test_pred)
        # print(y_test_pred)
        # print(y_test)
        onehot_encoder = OneHotEncoder(categories='auto').fit(y_test)
        y_test = onehot_encoder.transform(y_test).toarray().astype(np.bool)
        auc = roc_auc_score(y_test, y_pred_auc, average = None)
        auc_weighted = roc_auc_score(y_test, y_pred_auc, average = 'weighted')

            # auc = metrics.auc(fpr, tpr)

    print({
        'F1Mi': micro,
        'F1Ma': macro,
        'Recall': recall,
        'Precision': precision,
        'classification_report': cm,
        'AUC': auc,
        'AUC_weighted': auc_weighted
    })

            
    return eval_results, losses 
        
# def test_loader_mini4training(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
#     # data.y is labels of shape (N, ) 
    
#     model.eval()
#     data.x = data.x.to(device)
#     data.edge_index = data.edge_index.to(device)
#     data.y = data.y.to(device)
#     with torch.no_grad():
#         # print('xxxxxxxxxx',data.x)

#         data.x = model.encoder.inference(data.x, layer_loader, device)
#         data.x = data.x.detach().cpu().numpy()
#         data.x = torch.tensor(data.x).to(device)
#     # print('xxxxxxxxxx',data.x)
    
#     para_dict = sage_parameters
#     model_para = sage_parameters.copy()
#     model_para.pop('lr')
#     model_para.pop('l2')   
#     detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
#     optimizer = torch.optim.Adam(detect_model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
#     loss_function = torch.nn.CrossEntropyLoss()
#     best_valid = 0
#     min_valid_loss = 1e8
#     best_out = None
    
#     split_idx = {'train':data.train_mask.cpu(), 'valid':data.valid_mask.cpu(), 'test':data.test_mask.cpu()}
#     train_idx = split_idx['train'].to(device)
#     evaluator = evaluator('auc')
#     # print('xxxxxxxxxx222222',data.x)
    
#     for epoch in range(1, 1200+1):
#         # data.y = torch.randint(0,2, (data.y.size(0),)).to(device)
#         # data.x = torch.rand(data.x.size(0),data.x.size(1)).to(device)
#         detect_model.train()

#         optimizer.zero_grad()
#         # print('xxxxxxxxxx33333333333',data.x)
#         if no_conv:

#             print('data_x.size',data.x[train_idx].size())
#             out = detect_model(data.x[train_idx])
#         else:
#             print('data_x.size',data.x.size())
#             out = detect_model(data.x, data.edge_index)[train_idx]

#         loss = loss_function(out, data.y[train_idx])
#         # loss = F.nll_loss(out, data.y[train_idx])
#         loss.backward()
#         optimizer.step()
    
#         total_loss = loss.item()
#         eval_results, losses, out = gnn_test(detect_model, data, split_idx, evaluator, no_conv, is_loader = True)
#         train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
#         train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

#         if valid_loss < min_valid_loss:
#             min_valid_loss = valid_loss
#             best_out = out.cpu()
#             train_best_eval, valid_best_eval, test_best_eval = train_eval, valid_eval, test_eval
#         if epoch % 1 == 0:
#             print(f'Epoch: {epoch:02d}, '
#                 f'Loss: {total_loss:.4f}, '
#                 f'Train: {100 * train_eval:.3f}%, '
#                 f'Valid: {100 * valid_eval:.3f}% '
#                 f'Test: {100 * test_eval:.3f}%')
#         print(f'Best_Train: {100 * train_best_eval:.3f}%, '
#                 f'Best_Valid: {100 * valid_best_eval:.3f}% '
#                 f'Best_Test: {100 * test_best_eval:.3f}%')

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
    # device = 'cpu'

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
        delattr(data, 'adj_t')
        # data.edge_index = dataset.process().edge_index
        data = dataset.process()
        # data.edge_attr = torch.tensor(data.edge_attr, dtype=torch.long)
        data = data_process(data)
        # print(data.edge_attr, data.edge_attr.type())
        print('xsizeeeee',data.x.size())
        
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
        
    
    # data = data.to(device)
    # print('xxxxx11111',data.x)
    
    # train_idx = split_idx['train'].to(device)
    # batch_size = 8192
    batch_size = 512
    
    # train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[3,10], batch_size=batch_size, shuffle=True, num_workers=12)
    # layer_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=12) 

    # cluster_data = ClusterData(data, num_parts = 20000, recursive = False, save_dir = dataset.processed_dir)
    # sampler = ImbalancedSampler(data, input_nodes=train_idx)
    # sampler = ImbalancedSampler(data)

    # train_loader = ClusterLoader(cluster_data, batch_size=batch_size, shuffle=True,
    #                          num_workers=24)
    # train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2,
    #                                  num_steps=data.x.size(0)//batch_size//3+1, sample_coverage=10,
    #                                  save_dir=dataset.processed_dir,
    #                                  num_workers=20)
    # train_loader = RandomNodeLoader(data,num_parts = 200, num_workers=24)
    # train_loader = GraphSAINTNodeSampler(data, batch_size=batch_size, 
    #                                  num_steps=data.x.size(0)//batch_size+1, sample_coverage=10,
    #                                  save_dir=dataset.processed_dir,
    #                                  num_workers=24)
    # train_loader = GraphSAINTRandomWalkSampler(data, batch_size=batch_size, walk_length=2)
    # train_loader = ShaDowKHopSampler(data, depth=2, num_neighbors=25,
    #                             node_idx=train_idx, batch_size=batch_size, num_workers=0)
    train_loader = NeighborLoader(data, input_nodes=train_idx, num_neighbors=[10,10], batch_size=batch_size, num_workers=10)
    # train_loader = NeighborLoader(data, num_neighbors=[25,10], batch_size=batch_size, num_workers=24, sampler=sampler)
    layer_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=2048, shuffle=False, num_workers=24) 
    print('---------------------------------------------------------')
#     for item in train_loader:
#         print('kadsfjk',item)
    
    
    # print(data_edge_index.size,data_x.size())
    encoder = Encoder(data.x.size(-1), num_hidden, activation,base_model=base_model, k=num_layers).to(device)
    # print(dataset.num_features)
    model = Model(encoder, num_hidden, num_proj_hidden, tau).to(device)
    # model.reset_parameters()
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
#             torch.save(model, "best_embedding.pt")

    print("=== Final ===")

#     test(model, data.x, data.edge_index, data.y, final=True)
    
    if args.dataset == "DGraphFin":
        # test_mini(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
        # test(model, data.x.to(device), data.edge_index.to(device), data.y.to(device), final=True)
        # test_loader_mini4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
        test_mini_loader_4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    else:
        test(model, data.x.to(device), data.edge_index.to(device), data.y.to(device), final=True)

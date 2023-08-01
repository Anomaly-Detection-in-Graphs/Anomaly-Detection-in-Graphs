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
import copy

import torch
import torch_geometric.transforms as T
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.datasets import Planetoid, CitationFull
from torch_geometric.utils import dropout_adj, shuffle_node, dropout_node
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.nn import GraphConv, ResGatedGraphConv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, DNAConv, SignedConv, GCN2Conv, GENConv, ClusterGCNConv, SuperGATConv, EGConv, GeneralConv, GravNetConv, APPNP, FiLMConv, WLConvContinuous
from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader, RandomNodeLoader
from torch_geometric.loader import  NeighborSampler as NeighborSampler2
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report, roc_auc_score
from accelerate import Accelerator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize, OneHotEncoder

from models.gcl_neighsampler import Encoder, Model, drop_feature
from eval import label_classification, mlp_label_classification

from utils import DGraphFin as DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, SAGE_NeighSampler, GAT, GATv2
from logger import Logger
from gnn import train as gnn_train
from gnn import test as gnn_test
from feature_processing import data_process
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import time


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
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0
              , 'batchnorm': False
              , 'l2':5e-7
             }
accelerator = Accelerator()

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret



def test(model: Model, x, edge_index, y, final=False):
    model.eval()
    z = model(x, edge_index)

    label_classification(z, y, ratio=0.1)
            


def mini_sampler_4training(data, y, evaluator, device, args, no_conv=False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    # model.eval()
    
    eval_method = 'auc'

    data = data.to(device)
    print(data.x.size())
    
    # data.x = model.encoder.inference(data.x, layer_loader, device)
    # data.x = data.x.detach().cpu().numpy()
    # data.x = torch.tensor(data.x).to(device)
    
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')   
    detect_model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    # detect_model = SAGE(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
    optimizer = torch.optim.Adam(detect_model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    loss_function = torch.nn.CrossEntropyLoss()
    n_samples = data.x.shape[0]
    
    if args.dataset == "DGraphFin":
        split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}
        train_idx = split_idx['train'].to(device)
        print(torch.max(data.y[train_idx]))
        batch_size= 8196
        train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[10, 10], batch_size=batch_size, shuffle=True, num_workers=18)
        eval_train_loader = NeighborSampler(data.adj_t, node_idx=train_idx, sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
        eval_loader = NeighborSampler(data.adj_t, node_idx=split_idx['valid'].to(device), sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
        test_loader = NeighborSampler(data.adj_t, node_idx=split_idx['test'].to(device), sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
        
        
    else:
        ratio = 0.1
        valid_size = int(n_samples*ratio)
        test_size = int(n_samples*ratio)
        train_size = n_samples - valid_size - test_size
        indices = np.arange(n_samples)
        data_train, data_test, labels_train, labels_test, indices_train, indices_test = train_test_split(data.x, data.y, indices, test_size=test_size)
        data_train, data_val, labels_train, labels_val, indices_train, indices_val = train_test_split(data_train, labels_train, indices_train, test_size=valid_size)
        split_idx = {'train':torch.from_numpy(indices_train), 'valid':torch.from_numpy(indices_val), 'test':torch.from_numpy(indices_test)}
    
        train_idx = split_idx['train'].to(device)
        print(torch.max(data.y[train_idx]))
        batch_size= 8196
        train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[10, 10], batch_size=batch_size, shuffle=True, num_workers=18)
        eval_train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
        eval_loader = NeighborSampler(data.edge_index, node_idx=split_idx['valid'].to(device), sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
        test_loader = NeighborSampler(data.edge_index, node_idx=split_idx['test'].to(device), sizes=[-1, -1], batch_size=batch_size, shuffle=False, num_workers=18)
    
    best_valid = 0
    min_valid_loss = 1e12
    
    
    evaluator = evaluator(eval_method)
    counter = 0
    for epoch in range(1, 700+1):
        detect_model.train()
        pred = []
        total_loss = 0
        
        start_time = time.perf_counter()
        for batch_size, n_id, adjs in train_loader:
            
            
            # if counter == 1:
            #     start_time = time.perf_counter()
            adjs = [adj.to(device) for adj in adjs]
            sampled_data = data.x[n_id]
            # sampled_adjs = data.adj_t[n_id]
            
            y_true = data.y[n_id[:batch_size]]
            optimizer.zero_grad()
            # sampled_data.y = torch.nn.functional.one_hot(sampled_data.y).to(device)
            if no_conv:
                # print('data_x.size',sampled_data.x.size())
                out = detect_model(sampled_data)
            else:
                # out = detect_model(sampled_data.x, sampled_data.edge_index)
                out = detect_model(sampled_data, adjs)
                
            
            
            loss = loss_function(out, y_true)
            # loss = F.nll_loss(out, data.y[train_idx])
            
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
             
         
        print('evalllllllllllllllllllllllll', counter)
        counter += 1
        valid_loss = 0
        detect_model.eval()
        for batch_size, n_id, adjs in eval_loader:
            adjs = [adj.to(device) for adj in adjs]
            sampled_data = data.x[n_id]
            out = detect_model(sampled_data, adjs)
            valid_loss += loss_function(out, data.y[n_id[:batch_size]]).item()
        if counter > 40:
            break
            

        if valid_loss < min_valid_loss:
            counter = 0
            min_valid_loss = valid_loss
            eval_results, losses = gnn_test_loader(detect_model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, no_conv, is_loader=False)
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
        
##___________________________________________________________________________________________________
        
def mini_loader_4training(data, evaluator, device, args, no_conv=False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    # model.eval()
    
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
            eval_results, losses = gnn_test_loader(detect_model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, no_conv, is_loader=False)
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
def gnn_test_loader(model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, no_conv=False, is_loader=False):
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

@torch.no_grad()
def gnn_test_sampler(model, data, eval_train_loader, eval_loader, test_loader, split_idx, evaluator, no_conv=False, is_loader=False):
    # data.y is labels of shape (N, )
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()

    losses, eval_results = dict(), dict()
    train_pred, eval_pred, test_pred = [], [], []
    losses['train'], losses['valid'], losses['test'] = 0, 0, 0
    eval_method = 'auc'
    
    node_id = split_idx['train']
    for batch_size, n_id, adjs in eval_train_loader:
        adjs = [adj.to(device) for adj in adjs]
        sampled_data = data.x[n_id]
        # .to(device)
        out = model(sampled_data, adjs)
        losses['train'] += loss_function(out, data.y[n_id[:batch_size]].to(device)).item()
        train_pred.append(out.detach().cpu().exp())
    y_train_pred = torch.cat(train_pred, dim=0)
    eval_results['train'] = evaluator.eval(data.y[node_id], y_train_pred)[eval_method]
    
    
    node_id = split_idx['valid']
    for batch_size, n_id, adjs in eval_loader:
        adjs = [adj.to(device) for adj in adjs]
        sampled_data = data.x[n_id]
        # .to(device)
        out = model(sampled_data, adjs)
        losses['valid'] += loss_function(out, data.y[n_id[:batch_size]].to(device)).item()
        # out = out.exp()
        eval_pred.append(out.detach().cpu().exp())
    y_eval_pred = torch.cat(eval_pred, dim=0)
    eval_results['valid'] = evaluator.eval(data.y[node_id], y_eval_pred)[eval_method]
    
    node_id = split_idx['test']
    for batch_size, n_id, adjs in test_loader:
        adjs = [adj.to(device) for adj in adjs]
        sampled_data = data.x[n_id]
        # .to(device)
        # n_id = n_id.to(device)
        out = model(sampled_data, adjs)
        losses['test'] += loss_function(out, data.y[n_id[:batch_size]].to(device)).item()
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='PubMed')
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
    activation = ({'relu': nn.ReLU(), 'prelu': nn.PReLU(), 'elu': nn.ELU()})[config['activation']]
    base_model = ({'GCNConv': GCNConv, 'SAGEConv': SAGEConv, 'GATConv': GATConv, 'GATv2Conv': GATv2Conv, 'TAGConv': TAGConv, 'ResGatedGraphConv': ResGatedGraphConv, 'GraphConv': GraphConv, 'GeneralConv': GeneralConv, 'GENConv': GENConv, 'GravNetConv': GravNetConv, 'FiLMConv': FiLMConv, 'WLConvContinuous': WLConvContinuous})[config['base_model']]
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
        sim_filter_threshold = config['threshold']
        drop_node_rate_1 = config['drop_node_rate_1']
        drop_node_rate_2 = config['drop_node_rate_2']
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
        sim_filter_threshold = config['threshold']
        dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = 2
        args.nlabels = nlabels
        # print('dataset:', dataset)
        DGraphFin_data = dataset[0]
        DGraphFin_data = dataset.process()
        delattr(DGraphFin_data, 'adj_t')
        delattr(DGraphFin_data, 'edge_attr')
        
        # print('data:', data)
        # print('data.edge_index', data.edge_index)
        x = DGraphFin_data.x
        x = (x-x.mean(0))/x.std(0)
        DGraphFin_data.x = x
        if DGraphFin_data.y.dim()==2:
            DGraphFin_data.y = DGraphFin_data.y.squeeze(1)        
        
        split_idx = {'train':DGraphFin_data.train_mask, 'valid':DGraphFin_data.valid_mask, 'test':DGraphFin_data.test_mask}
        train_idx = DGraphFin_data.train_mask
        # .to(device)
        
        path = osp.join(osp.expanduser('~'), 'datasets', 'Cora')
        dataset = get_dataset(path, 'Cora')        
        data = dataset[0]
        data.x = DGraphFin_data.x
        data.y = DGraphFin_data.y
        data.edge_index = DGraphFin_data.edge_index
        data.train_mask = DGraphFin_data.train_mask
        data.valid_mask = DGraphFin_data.valid_mask
        data.test_mask = DGraphFin_data.test_mask
        
        print('lengthhhhh',len(train_idx))
    
    data = data.to(device)
    
    # train_idx = split_idx['train'].to(device)
   
    
    # print(data_edge_index.size,data.x.size())
    dataset_num_features = dataset.num_features

    if args.dataset == "DGraphFin":
        is_DGraph = True
    else:
        is_DGraph = False
    mini_loss = 1*10000

#     test(model, data.x, data.edge_index, data.y, final=True)
    # test_mini(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    # test_mini4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    # test_mini_loader_4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    mini_loader_4training(data, Evaluator, device, args, no_conv=False, use_mlp = True)

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
from torch_geometric.utils import dropout_adj, shuffle_node, dropout_node
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GATv2Conv
from torch_geometric.nn import GraphConv, ResGatedGraphConv, TransformerConv, AGNNConv, TAGConv, GINConv, GINEConv, ARMAConv, SGConv, DNAConv, SignedConv, GCN2Conv, GENConv, ClusterGCNConv, SuperGATConv, EGConv, GeneralConv, GravNetConv, APPNP, FiLMConv, WLConvContinuous
from torch_geometric.data import NeighborSampler
from torch_geometric.loader import NeighborLoader, LinkNeighborLoader
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report, roc_auc_score
from accelerate import Accelerator
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import normalize, OneHotEncoder

from models.gcl_neighsampler import Encoder, Model, drop_feature
from eval import label_classification, mlp_label_classification

from utils import DGraphFin2 as DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, SAGE_NeighSampler, GAT, GATv2
from logger import Logger
from gnn import train as gnn_train
from gnn import test as gnn_test
from feature_processing import data_process

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

def train(epoch, train_loader, model, data, train_idx, optimizer, device, rate_dic, no_conv=False, is_DGraph = False, batch_size = 0):
    model.train()

    pbar = tqdm(total=train_idx.size(0), ncols=80)
    pbar.set_description(f'Epoch {epoch:02d}')

    total_loss = total_correct = 0
    for batch_size, n_id, adjs in train_loader:
#     for sampled_hetero_data in train_loader:
#         batch_size = sampled_hetero_data.batch_size
        
#         x_1 = drop_feature(sampled_hetero_data.x, drop_feature_rate_1)
#         x_2, _ = shuffle_node(sampled_hetero_data.x)
        

        adjs = [adj.to(device) for adj in adjs]
        # for edge_index, batch, size in adjs:
        #     print('edge size', size)
        #     print('size', data.x[n_id].size())
        #     print('edge size', edge_index.size())
#         adjs_1 = [(dropout_adj(edge_index, p=rate_dic['drop_edge_rate_1'])[0], batch, size) for edge_index, batch, size in adjs]
#         adjs_2 = [(dropout_adj(edge_index, p=rate_dic['drop_edge_rate_2'])[0], batch, size) for edge_index, batch, size in adjs]
        
#         x_1 = drop_feature(data.x[n_id], rate_dic['drop_feature_rate_1'])
#         x_2 = drop_feature(data.x[n_id], rate_dic['drop_feature_rate_2'])
#         # x_2, _ = shuffle_node(data.x[n_id])
     
        x_1 = []
        x_2 = []
        adjs_1 = []
        adjs_2 = []
        
        for edge_index, batch, size in adjs:
            edge_index, edge_mask, node_mask = dropout_node(edge_index, p= rate_dic['drop_node_rate_1'])
            # masked_indices = np.where(node_mask)[0]
            # masked_nodes = n_id[~np.isin(n_id, masked_indices)]
            x_1.append(drop_feature(data.x[n_id], rate_dic['drop_feature_rate_1']))
            adjs_1.append((edge_index, batch, size))
            
            edge_index, edge_mask, node_mask = dropout_node(edge_index, p= rate_dic['drop_node_rate_2'])
            # masked_indices = np.where(node_mask)[0]
            # masked_nodes = n_id[~np.isin(n_id, masked_indices)]
            x_2.append(drop_feature(data.x[n_id], rate_dic['drop_feature_rate_2']))
            adjs_2.append((edge_index, batch, size))
            
        # x_1 = [(dropout_node(edge_index, p=drop_node_rate_1)[0], batch, size) for edge_index, batch, size in adjs]
        # x_2 = [(dropout_node(edge_index, p=drop_node_rate_2)[0], batch, size) for edge_index, batch, size in adjs]
        
        
        
        
        # print(batch_size, n_id)
        # print('adj_index',adjs[0][0].size())
        # print('_',adjs[0][1].size())
        # print('size',adjs[0][2])
        optimizer.zero_grad()
        z1 = model(x_1, adjs_1)
        z2 = model(x_2, adjs_2) 
        loss = model.loss(z1, z2, batch_size=batch_size, cur_epoch = epoch)
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
    # data.y is labels of shape (N, ) 
    model.eval()
    
    model.eval()
    data.x = data.x.to(device)
    data.edge_index = data.edge_index.to(device)
    data.y = data.y.to(device)
    with torch.no_grad():
        # print('xxxxxxxxxx',data.x)

        data.x = model.encoder.inference(data.x, layer_loader, device)
        data.x = data.x.detach().cpu().numpy()
        data.x = torch.tensor(data.x).to(device)
    
    # data.x = model.encoder.inference(data.x, layer_loader, device)
    # data.x = data.x.detach().cpu().numpy()
    # data.x = torch.tensor(data.x).to(device)
    
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
    n_samples = data.shape[0]
    eval_method = 'auc'
    
    if args.dataset == "DGraphFin":
        split_idx = {'train':data.train_mask.cpu(), 'valid':data.valid_mask.cpu(), 'test':data.test_mask.cpu()}
        train_idx = split_idx['train'].to(device)
        evaluator = evaluator(eval_method)
    else:
        ratio = 0.1
        valid_size = n_samples//(1/0.1)
        test_size = n_samples//(1/0.1)
        train_size = n_samples - valid_size - test_size
        indices = np.arange(n_samples)
        data_train, data_test, labels_train, labels_test, indices_train, indices_test = train_test_split(data.x, data.y, indices, test_size=test_size)
        data_train, data_val, labels_train, labels_val, indices_train, indices_val = train_test_split(data_train, labels_train, indices_train, test_size=valid_size)
        split_idx = {'train':indices_train, 'valid':indices_val, 'test':indices_test}
        train_idx = split_idx['train'].to(device)
        evaluator = evaluator(eval_method)
        
    
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
        
    with open('edge_droping.txt', 'a') as f:
        f.write(f'Node Drop Rate 1: {args.drop_edge_rate_1}, '
               f'Node Drop Rate 2: {args.drop_edge_rate_2}, ')
        f.write(f'Best_Train: {100 * train_best_eval:.3f}%, '
                f'Best_Valid: {100 * valid_best_eval:.3f}% '
                f'Best_Test: {100 * test_best_eval:.3f}%')
        

def test_mini_loader_4training(layer_loader, model, data, y, evaluator, device, args, no_conv=False, use_mlp = False):
    # data.y is labels of shape (N, ) 
    model.eval()
    eval_method = 'auc'
    # device = 'cpu'
#     data = data.to(device)
#     model = model.to(device)
    
    # delattr(data, 'adj_t')
    with torch.no_grad():
        # print('xxxxxxxxxx',data.x)

        data.x = model.encoder.inference(data.x, layer_loader, device)
        data.x = data.x.detach().cpu().numpy()
        data.x = torch.tensor(data.x)
    data = data.to(device)
    
    # data.x = model.encoder.inference(data.x, layer_loader, device)
    # data.x = data.x.detach().cpu().numpy()
    # data.x = torch.tensor(data.x).to(device)
    
    para_dict = sage_parameters
    model_para = sage_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')   
    detect_model = SAGE_NeighSampler(in_channels = data.x.size(-1), out_channels = args.nlabels, **model_para).to(device)
    
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
    min_valid_loss = 1e8
    
    
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
    with open('node_droping.txt', 'a') as f:
        f.write(f'Node Drop Rate 1: {args.rate_1}, '
               f'Node Drop Rate 2: {args.rate_2} \n')
        f.write(f'Best_Train: {100 * train_best_eval:.3f}%, '
                f'Best_Valid: {100 * valid_best_eval:.3f}% '
                f'Best_Test: {100 * test_best_eval:.3f}% \n')

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


def main(drop_rate_1, drop_rate_2):
    parser = argparse.ArgumentParser()
    # parser.add_argument('--dataset', type=str, default='DBLP')
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
    
    rate_dic = {}

    rate_dic['drop_edge_rate_1'] = config['drop_edge_rate_1']
    rate_dic['drop_edge_rate_2'] = config['drop_edge_rate_2']
    rate_dic['drop_feature_rate_1'] = config['drop_feature_rate_1']
    rate_dic['drop_feature_rate_2'] = config['drop_feature_rate_2']
    rate_dic['drop_node_rate_1'] = drop_rate_1 
    rate_dic['drop_node_rate_2'] = drop_rate_2
    args.rate_1 = drop_rate_1
    args.rate_2 = drop_rate_2
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
        # drop_node_rate_1 = config['drop_node_rate_1']
        # drop_node_rate_2 = config['drop_node_rate_2']
        path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
        dataset = get_dataset(path, args.dataset)
        nlabels = dataset.num_classes
        args.nlabels = nlabels
        
        data = dataset[0]
        train_idx = np.array(list(range(len(data.x))))
        train_idx = torch.LongTensor(train_idx).to(device)
        # print("dataset:", dataset,dataset[0], len(dataset))
    elif args.dataset == "DGraphFin":
        # drop_node_rate_1 = config['drop_node_rate_1']
        # drop_node_rate_2 = config['drop_node_rate_2']
        
        sim_filter_threshold = config['threshold']
        dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = 2
        args.nlabels = nlabels
        # print('dataset:', dataset)
        data = dataset[0]
        # print('data:', data)
        # print('data.edge_index', data.edge_index)
        data.edge_index = dataset.process().edge_index
        data = data_process(data)
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
        # train_idx = torch.cat((data.train_mask, data.valid_mask, data.test_mask), dim=0)
        train_idx = data.train_mask
        # .to(device)
        print('lengthhhhh',len(train_idx))
    
    data = data.to(device)
    
    # train_idx = split_idx['train'].to(device)
    batch_size = 2048
    
    train_loader = NeighborSampler(data.edge_index, node_idx=train_idx, sizes=[3, 10], batch_size=batch_size, shuffle=True, num_workers=24)
    layer_loader = NeighborSampler(data.edge_index, node_idx=None, sizes=[-1], batch_size=4096, shuffle=False, num_workers=24) 

    
    # print(data_edge_index.size,data_x.size())
    dataset_num_features = dataset.num_features
    # print('numnumnumnum', dataset_num_features)
    encoder = Encoder(data.x.size(-1), num_hidden, activation,base_model=base_model, k=num_layers).to(device)
    # print(dataset.num_features)
    model = Model(encoder, num_hidden, num_proj_hidden, tau, sim_filter_threshold = sim_filter_threshold).to(device)
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
    for epoch in range(1, num_epochs + 1):
        loss = train(epoch, train_loader, model, data, train_idx, optimizer, device, rate_dic, no_conv=False, batch_size = batch_size, is_DGraph = is_DGraph)

        now = t()
        print(f'(T) | Epoch={epoch:03d}, loss={loss:.4f}, '
              f'this epoch {now - prev:.4f}, total {now - start:.4f}')
        prev = now
        if loss < mini_loss:
            mini_loss = loss
            torch.save(model, "best_embedding.pt")

    print("=== Final ===")

#     test(model, data.x, data.edge_index, data.y, final=True)
    # test_mini(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    # test_mini4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    test_mini_loader_4training(layer_loader, model, data, data.y, Evaluator, device, args, no_conv=False, use_mlp = True)
    
if __name__ == '__main__':
    device = 'cuda'
    
    drop_rate_1 = [0.0, 0.01, 0.02, 0.03, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]
    drop_rate_2 = [0.25, 0.25, 0.25, 0.25, 0.225, 0.2, 0.175, 0.15, 0.125, 0.1]
    for rate_1, rate_2 in zip(drop_rate_1, drop_rate_2):
        main(rate_1, rate_2)
    

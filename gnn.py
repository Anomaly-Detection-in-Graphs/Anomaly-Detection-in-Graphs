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
# from logger import Logger
from feature_processing import data_process
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' 

import time
import warnings
warnings.filterwarnings("ignore")


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

def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret

def get_dataset(path, name):
    assert name in ['Cora', 'CiteSeer', 'PubMed', 'DBLP']
    name = 'dblp' if name == 'DBLP' else name
    return (CitationFull if name == 'dblp' else Planetoid)(
            path,
            name,
            transform=T.NormalizeFeatures())

def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    if no_conv:
        
#         print('data_x.size',data.x[train_idx].size())
        out = model(data.x[train_idx])
    else:
        # print('data_adj.size',data.adj_t.size)
        # print('data_x.size',data.x.size())
        # out = model(data.x, data.edge_index)[train_idx]
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, acc_evaluator, auc_evaluator, args, no_conv=False, is_loader=False):
    # data.y is labels of shape (N, )
    model.eval()
    loss_function = torch.nn.CrossEntropyLoss()
    
    if no_conv:
        out = model(data.x)
    else:
        if is_loader:
            out = model(data.x, data.edge_index)
        else:
            out = model(data.x, data.adj_t)
        
    y_pred = out.exp()  # (N,num_classes)
    # print('yyyyyyyyyyyyyyyyy',y_pred)
    
    losses, eval_results = dict(), {'acc': dict(), 'auc': dict()}
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        # losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        losses[key] = loss_function(out[node_id], data.y[node_id]).item()
        eval_results['acc'][key] = acc_evaluator.eval(data.y[node_id], y_pred[node_id])['acc']
        eval_results['auc'][key] = auc_evaluator.eval(data.y[node_id], y_pred[node_id])['auc']
    
    #------------------------------------------------------

    
    
    
    #------------------------------------------------------
    
    if args.dataset != "DGraphFin":
        y_test_pred = out.detach().cpu().exp()[split_idx['test']]
        y_discrete_pred = y_test_pred.argmax(axis=-1)

        sorted_y = np.array(y_discrete_pred)
        # print('outoutout', np.sort(sorted_y))
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
        onehot_encoder = OneHotEncoder(categories='auto').fit(y_test)
        y_test = onehot_encoder.transform(y_test).toarray().astype(np.bool)
        auc = roc_auc_score(y_test, y_pred_auc, average = None)
        auc_weighted = roc_auc_score(y_test, y_pred_auc, average = 'weighted')
        
        
    elif args.dataset == "DGraphFin":
        y_discrete_pred = out.detach().cpu().numpy()
        y_discrete_pred = y_discrete_pred[split_idx['test']].argmax(axis=-1)
        # sorted_y, _ = torch.sort(y_discrete_pred, descending=True)
        sorted_y = np.array(y_discrete_pred)
        # print('outoutout', np.sort(sorted_y))
        y_pred_auc = out.detach().cpu().numpy()
        y_pred_auc = y_pred_auc[split_idx['test']]
        y_pred_auc = y_pred_auc[:, 1]
        y_test = data.y[split_idx['test']].detach().cpu().numpy()
        y_test = y_test.reshape(-1, 1)

        micro = f1_score(y_test, y_discrete_pred, average="micro")
        macro = f1_score(y_test, y_discrete_pred, average="macro")
        recall = recall_score(y_test, y_discrete_pred, labels = [0,1], average=None)
        precision = precision_score(y_test, y_discrete_pred, labels = [0,1], average=None)
        cm = classification_report(y_test, y_discrete_pred)
        auc = roc_auc_score(y_test, y_pred_auc, average = None)
        auc_weighted = roc_auc_score(y_test, y_pred_auc, average = 'weighted')
    
    # auc = metrics.auc(fpr, tpr)

    # print({
    #     'F1Mi': micro,
    #     'F1Ma': macro,
    #     'Recall': recall,
    #     'Precision': precision,
    #     'classification_report': cm,
    #     'AUC': auc,
    #     'AUC_weighted': auc_weighted
    # })
            
    return eval_results, losses, y_pred
 
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='CiteSeer')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='sage')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=700)
    parser.add_argument('--runs', type=int, default=1)
    parser.add_argument('--fold', type=int, default=0)
    
    args = parser.parse_args()
    print(args)
    
    no_conv = False
    if args.model in ['mlp']: no_conv = True        
    
    device = f'cuda:{args.device}' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    
        
    
        
    if args.dataset != "DGraphFin":
        path = osp.join(osp.expanduser('~'), 'datasets', args.dataset)
        dataset = get_dataset(path, args.dataset)
        nlabels = dataset.num_classes
        args.nlabels = nlabels
        
        data = dataset[0]
        data.adj_t = data.edge_index
        train_idx = np.array(list(range(len(data.x))))
        train_idx = torch.LongTensor(train_idx).to(device)
        
        n_samples = data.x.shape[0]
        ratio = 0.1
        valid_size = int(n_samples*ratio)
        test_size = int(n_samples*ratio)
        train_size = n_samples - valid_size - test_size
        indices = np.arange(n_samples)
        data_train, data_test, labels_train, labels_test, indices_train, indices_test = train_test_split(data.x, data.y, indices, test_size=test_size)
        data_train, data_val, labels_train, labels_val, indices_train, indices_val = train_test_split(data_train, labels_train, indices_train, test_size=valid_size)
        split_idx = {'train':torch.from_numpy(indices_train), 'valid':torch.from_numpy(indices_val), 'test':torch.from_numpy(indices_test)}
        
        # print("dataset:", dataset,dataset[0], len(dataset))
    elif args.dataset == "DGraphFin":
        dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
        
        nlabels = 2
        args.nlabels = nlabels
        # print('dataset:', dataset)
        data = dataset[0]
        data.adj_t = data.adj_t.to_symmetric()
        
        # print('data:', data)
        # print('data.edge_index', data.edge_index)
        x = DGraphFin_data.x
        x = (x-x.mean(0))/x.std(0)
        DGraphFin_data.x = x
        if DGraphFin_data.y.dim()==2:
            DGraphFin_data.y = DGraphFin_data.y.squeeze(1)        
        
        split_idx = {'train':DGraphFin_data.train_mask, 'valid':DGraphFin_data.valid_mask, 'test':DGraphFin_data.test_mask}
        train_idx = DGraphFin_data.train_mask
        
        x = data.x
        x = (x-x.mean(0))/x.std(0)
        data.x = x
        if data.y.dim()==2:
            data.y = data.y.squeeze(1)        
    
        split_idx = {'train':data.train_mask, 'valid':data.valid_mask, 'test':data.test_mask}

    fold = args.fold
    if split_idx['train'].dim()>1 and split_idx['train'].shape[1] >1:
        kfolds = True
        print('There are {} folds of splits'.format(split_idx['train'].shape[1]))
        split_idx['train'] = split_idx['train'][:, fold]
        split_idx['valid'] = split_idx['valid'][:, fold]
        split_idx['test'] = split_idx['test'][:, fold]
    else:
        kfolds = False
        
    data = data.to(device)
    train_idx = split_idx['train'].to(device)
        
    result_dir = prepare_folder(args.dataset, args.model)
    print('result_dir:', result_dir)
        
    if args.model == 'mlp':
        para_dict = mlp_parameters
        model_para = mlp_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')
        model = MLP(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'gcn':   
        para_dict = gcn_parameters
        model_para = gcn_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = GCN(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)
    if args.model == 'sage':        
        para_dict = sage_parameters
        model_para = sage_parameters.copy()
        model_para.pop('lr')
        model_para.pop('l2')        
        model = SAGE(in_channels = data.x.size(-1), out_channels = nlabels, **model_para).to(device)

    print(f'Model {args.model} initialized')

    acc_evaluator = Evaluator('acc')
    auc_evaluator = Evaluator('auc')
    # logger = Logger(args.runs, args)

    for run in range(args.runs):
        import gc
        gc.collect()
        # print(sum(p.numel() for p in model.parameters()))

        model.reset_parameters()
        optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
        best_valid = 0
        min_valid_loss = 1e8
        best_out = None

        for epoch in range(1, args.epochs+1):
            loss = train(model, data, train_idx, optimizer, no_conv)
            eval_results, losses, out = test(model, data, split_idx, acc_evaluator, auc_evaluator, args, no_conv)
            acc_train_eval, acc_valid_eval, acc_test_eval = eval_results['acc']['train'], eval_results['acc']['valid'], eval_results['acc']['test']
            auc_train_eval, auc_valid_eval, auc_test_eval = eval_results['auc']['train'], eval_results['auc']['valid'], eval_results['auc']['test']
            train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

#                 if valid_eval > best_valid:
#                     best_valid = valid_result
#                     best_out = out.cpu().exp()
            if valid_loss < min_valid_loss:
                min_valid_loss = valid_loss
                best_out = out.cpu()
                acc_train_best_eval, acc_valid_best_eval, acc_test_best_eval = acc_train_eval, acc_valid_eval, acc_test_eval
                auc_train_best_eval, auc_valid_best_eval, auc_test_best_eval = auc_train_eval, auc_valid_eval, auc_test_eval

                print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                          f'Acc Train: {100 * acc_train_best_eval:.3f}%, '
                          f'Acc Valid: {100 * acc_valid_best_eval:.3f}% '
                          f'Acc Test: {100 * acc_test_best_eval:.3f}%'
                         f'Auc Train: {100 * auc_train_best_eval:.3f}%, '
                          f'Auc Valid: {100 * auc_valid_best_eval:.3f}% '
                      f'Auc Test: {100 * auc_test_best_eval:.3f}%')
#             logger.add_result(run, [train_eval, valid_eval, test_eval])

#         logger.print_statistics(run)

#     final_results = logger.print_statistics()
    # print('final_results:', final_results)
    # para_dict.update(final_results)
    # pd.DataFrame(para_dict, index=[args.model]).to_csv(result_dir+'/results.csv')


if __name__ == "__main__":
    main()

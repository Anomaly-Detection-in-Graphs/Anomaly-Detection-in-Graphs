""" 
Arthur: @Mingyang Zhao
""" 


from utils import DGraphFin
from utils.utils import prepare_folder
from utils.evaluator import Evaluator
from models import MLP, MLPLinear, GCN, SAGE, GAT, GATv2
from logger import Logger
from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
import numpy as np
import dgl
from pyvis.network import Network
import networkx as nx

import argparse

import torch
import torch.nn.functional as F
import torch.nn as nn

import torch_geometric
import torch_geometric.transforms as T
from torch_sparse import SparseTensor
from torch_geometric.utils import to_undirected
import pandas as pd

eval_metric = 'auc'

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


def train(model, data, train_idx, optimizer, no_conv=False):
    # data.y is labels of shape (N, ) 
    model.train()

    optimizer.zero_grad()
    if no_conv:
        
        print('data_x.size',data.x[train_idx].size())
        out = model(data.x[train_idx])
    else:
        print('data_adj.size',data.adj_t.size)
        print('data_x.size',data.x.size())
        out = model(data.x, data.adj_t)[train_idx]
    loss = F.nll_loss(out, data.y[train_idx])
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def test(model, data, split_idx, evaluator, no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()
    
    if no_conv:
        out = model(data.x)
    else:
        out = model(data.x, data.adj_t)
        
    y_pred = out.exp()  # (N,num_classes)
    
    losses, eval_results = dict(), dict()
    for key in ['train', 'valid', 'test']:
        node_id = split_idx[key]
        losses[key] = F.nll_loss(out[node_id], data.y[node_id]).item()
        eval_results[key] = evaluator.eval(data.y[node_id], y_pred[node_id])[eval_metric]
    
    
    y_discrete_pred = out.detach().cpu().numpy()
    y_discrete_pred = y_discrete_pred[split_idx['test']].argmax(axis=-1)
    # sorted_y, _ = torch.sort(y_discrete_pred, descending=True)
    sorted_y = np.array(y_discrete_pred)
    print('outoutout', np.sort(sorted_y))
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

    print({
        'F1Mi': micro,
        'F1Ma': macro,
        'Recall': recall,
        'Precision': precision,
        'classification_report': cm,
        'AUC': auc,
        'AUC_weighted': auc_weighted
    })
            
    return eval_results, losses, y_pred
        
            
def main():
    parser = argparse.ArgumentParser(description='gnn_models')
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--dataset', type=str, default='DGraphFin')
    parser.add_argument('--log_steps', type=int, default=10)
    parser.add_argument('--model', type=str, default='sage')
    parser.add_argument('--use_embeddings', action='store_true')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--runs', type=int, default=10)
    parser.add_argument('--fold', type=int, default=0)
    
    args = parser.parse_args()
    print(args)


    dataset = DGraphFin(root='./dataset/', name=args.dataset, transform=T.ToSparseTensor())
    tensor_data = dataset[0]
    tensor_data.edge_index = dataset.process().edge_index
    print(tensor_data.adj_t)
    # tensor_data.adj_t = tensor_data.adj_t.to_dense()
    # detach().cpu().numpy()
    
    nlabels = dataset.num_classes
    if args.dataset in ['DGraphFin']: nlabels = 2
    
    g = Network(height=800, width=800, notebook=True)
    data = torch_geometric.data.Data(x=tensor_data.x, edge_index=tensor_data.edge_index)

    netxG = nx.Graph(torch_geometric.utils.to_networkx(data))

    mapping = {i:i for i in range(netxG.size())} #Setting mapping for the relabeling
    netxH = nx.relabel_nodes(netxG,mapping) #relabeling nodes
    
    # nx.draw_network(netxH)
    # plt.show()

    g.from_nx(netxH)
    g.save_graph('ex.html')
    # g.show('ex.html')
        


if __name__ == "__main__":
    main()

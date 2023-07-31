"""  
Arthur: @Zeyang Cui, Zihan Xie, Mingyang Zhao
""" 

import numpy as np
import functools

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix, multilabel_confusion_matrix, classification_report
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import normalize, OneHotEncoder
from models.mlp import Prob_Network, MLP
import torch
import torch.nn.functional as F
from utils.evaluator import Evaluator

mlp_parameters = {'lr':0.01
              , 'num_layers':2
              , 'hidden_channels':128
              , 'dropout':0.0
              , 'batchnorm': False
              , 'l2':5e-7
             }

def repeat(n_times):
    def decorator(f):
        @functools.wraps(f)
        def wrapper(*args, **kwargs):
            results = [f(*args, **kwargs) for _ in range(n_times)]
            statistics = {}
            for key in results[0].keys():
                values = [r[key] for r in results]
                statistics[key] = {
                    'mean': np.mean(values),
                    'std': np.std(values)}
            print_statistics(statistics, f.__name__)
            return statistics
        return wrapper
    return decorator


def prob_to_one_hot(y_pred):
    ret = np.zeros(y_pred.shape, np.bool)
    indices = np.argmax(y_pred, axis=1)
    for i in range(y_pred.shape[0]):
        ret[i][indices[i]] = True
    return ret


def print_statistics(statistics, function_name):
    print(f'(E) | {function_name}:', end=' ')
    for i, key in enumerate(statistics.keys()):
        mean = statistics[key]['mean']
        std = statistics[key]['std']
        print(f'{key}={mean:.4f}+-{std:.4f}', end='')
        if i != len(statistics.keys()) - 1:
            print(',', end=' ')
        else:
            print()

def mlp_label_classification(embeddings, y, split_idx):
    eval_metric = 'auc'
    evaluator = Evaluator(eval_metric)
    nlabels = 2
    device = f'cuda' if torch.cuda.is_available() else 'cpu'
    device = torch.device(device)
    embeddings = embeddings.detach().cpu().numpy()
    embeddings = torch.tensor(embeddings).to(device)
    # embeddings = embeddings.to(device)
    y = y.to(device)
    # y = torch.randint(0,2, (y.size(0),)).to(device)
    print(embeddings.size())
    
    
    if y.dim()==2:
        y = y.squeeze(1) 
    
    para_dict = mlp_parameters
    model_para = mlp_parameters.copy()
    model_para.pop('lr')
    model_para.pop('l2')
    model = MLP(in_channels = embeddings.size(-1), out_channels = nlabels, **model_para).to(device)

    # model = Prob_Network(in_channels = embeddings.size(-1), hidden_channels = 128, out_channels = nlabels).to(device)
    # model.reset_parameters()
    optimizer = torch.optim.Adam(model.parameters(), lr=para_dict['lr'], weight_decay=para_dict['l2'])
    loss_function = torch.nn.CrossEntropyLoss()
    best_valid = 0
    min_valid_loss = 1e8
    best_out = None
    
    for epoch in range(1, 750+1):
        model.train()

        optimizer.zero_grad()
        out = model(embeddings[split_idx['train']])
        # out = model(embeddings[0,:])
        # print(out)
        
        # loss = F.nll_loss(out, y[split_idx['train']])
        loss = loss_function(out, y[split_idx['train']])
        # print(y.size())
        # print(loss)
        loss.backward()
        
        optimizer.step()
        
        model.eval()
        out = model(embeddings)
        y_pred = out.exp()  # (N,num_classes)

        losses, eval_results = dict(), dict()
        for key in ['train', 'valid', 'test']:
            node_id = split_idx[key]
            losses[key] = loss_function(out[node_id], y[node_id]).item()
            eval_results[key] = evaluator.eval(y[node_id], y_pred[node_id])[eval_metric]
 
        train_eval, valid_eval, test_eval = eval_results['train'], eval_results['valid'], eval_results['test']
        train_loss, valid_loss, test_loss = losses['train'], losses['valid'], losses['test']

        if valid_loss < min_valid_loss:
            min_valid_loss = valid_loss
            best_out = out.cpu()

        if epoch % 2 == 0:
            print(f'Epoch: {epoch:02d}, '
                      f'Loss: {loss:.4f}, '
                      f'Train: {100 * train_eval:.3f}%, '
                      f'Valid: {100 * valid_eval:.3f}% '
                      f'Test: {100 * test_eval:.3f}%')
    

    y_pred = model(embeddings[split_idx['test']])
    y_pred = y_pred.detach().cpu().numpy()
    y_discrete_pred = y_pred.argmax(axis=-1)
    y_pred = y_pred[:, 1]
    y_test = y[split_idx['test']].detach().cpu().numpy()
    y_test = y_test.reshape(-1, 1)
    print(y_discrete_pred)
    print(y_test)

    micro = f1_score(y_test, y_discrete_pred, average="micro")
    macro = f1_score(y_test, y_discrete_pred, average="macro")
    recall = recall_score(y_test, y_discrete_pred, labels = [0,1], average=None)
    precision = precision_score(y_test, y_discrete_pred, labels = [0,1], average=None)
    cm = classification_report(y_test, y_discrete_pred)
    print(y_pred)
    auc = roc_auc_score(y_test, y_pred, average = None)
    auc_weighted = roc_auc_score(y_test, y_pred, average = 'weighted')
    # auc = metrics.auc(fpr, tpr)

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'Recall': recall,
        'Precision': precision,
        'classification_report': cm,
        'AUC': auc,
        'AUC_weighted': auc_weighted
    }

# @repeat(3)
def label_classification(embeddings, y, ratio):
    X = embeddings.detach().cpu().numpy()
    Y = y.detach().cpu().numpy()
    Y = Y.reshape(-1, 1)
    onehot_encoder = OneHotEncoder(categories='auto').fit(Y)
    Y = onehot_encoder.transform(Y).toarray().astype(np.bool)

    X = normalize(X, norm='l2')

    X_train, X_test, y_train, y_test = train_test_split(X, Y,
                                                        test_size=1 - ratio)

    # logreg = LogisticRegression(solver='liblinear')
    logreg = RandomForestClassifier()
    c = 2.0 ** np.arange(-10, 10)
    n_estimators= [100]

    # clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
    #                    param_grid=dict(estimator__C=c), n_jobs=8, cv=5,
    #                    verbose=0)
    clf = GridSearchCV(estimator=OneVsRestClassifier(logreg),
                       param_grid=dict(estimator__n_estimators=n_estimators), n_jobs=8, cv=5,
                       verbose=0)
    clf.fit(X_train, y_train)

    y_pred = clf.predict_proba(X_test)
    print(y_pred)
    y_pred = prob_to_one_hot(y_pred)
    print('sdfgdfsgsdf',y_test)

    micro = f1_score(y_test, y_pred, average="micro")
    macro = f1_score(y_test, y_pred, average="macro")
    recall = recall_score(y_test, y_pred, labels = [0,1], average=None)
    precision = precision_score(y_test, y_pred, labels = [0,1], average=None)
    cm = classification_report(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred, average = None)
    auc_weighted = roc_auc_score(y_test, y_pred, average = 'weighted')
    # auc = metrics.auc(fpr, tpr)

    return {
        'F1Mi': micro,
        'F1Ma': macro,
        'Recall': recall,
        'Precision': precision,
        'classification_report': cm,
        'AUC': auc,
        'AUC_weighted': auc_weighted
    }

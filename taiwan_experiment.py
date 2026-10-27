"""Evaluate thesis-inspired classifiers on the UCI Taiwanese bankruptcy dataset."""
import argparse
import copy
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import accuracy_score, average_precision_score, confusion_matrix, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from compare_models import RBFNetwork, HierarchicalFuzzy
from train import NeuroFuzzy

# Predefined financial indicators, selected without inspecting test labels.
FEATURES = ['Current Ratio', 'Quick Ratio', 'Debt ratio %', 'Net worth/Assets',
            'Working Capital to Total Assets', 'Current Assets/Total Assets',
            'Retained Earnings to Total Assets', 'Net Income to Total Assets']

class MLP(nn.Module):
    def __init__(self, n):
        super().__init__(); self.net=nn.Sequential(nn.Linear(n,16),nn.ReLU(),nn.Linear(16,1))
    def forward(self,x): return self.net(x).squeeze(1)

def train(model, xt, yt, xv, yv, seed):
    torch.manual_seed(seed)
    tx=torch.tensor(xt,dtype=torch.float32); ty=torch.tensor(yt,dtype=torch.float32)
    vx=torch.tensor(xv,dtype=torch.float32); vy=torch.tensor(yv,dtype=torch.float32)
    # Positive-class weighting affects optimization only; validation uses ordinary loss.
    weight=torch.tensor((yt==0).sum()/(yt==1).sum(),dtype=torch.float32)
    training_loss=nn.BCEWithLogitsLoss(pos_weight=weight); validation_loss=nn.BCEWithLogitsLoss()
    opt=torch.optim.Adam(model.parameters(),lr=.01,weight_decay=.001)
    best=float('inf'); best_epoch=0; state=None
    for epoch in range(1,401):
        model.train(); opt.zero_grad(); loss=training_loss(model(tx),ty); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): current=validation_loss(model(vx),vy).item()
        if current < best-1e-5: best=current; best_epoch=epoch; state=copy.deepcopy(model.state_dict())
        if epoch-best_epoch>=50: break
    model.load_state_dict(state); model.eval()
    return model,best_epoch

def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('file'); parser.add_argument('--seed',type=int,default=42); a=parser.parse_args()
    torch.set_num_threads(1); df=pd.read_csv(a.file); df.columns=df.columns.str.strip()
    missing=set(FEATURES+['Bankrupt?'])-set(df.columns)
    if missing: parser.error('Missing columns: '+', '.join(sorted(missing)))
    x=df[FEATURES].to_numpy(dtype=float); y=df['Bankrupt?'].to_numpy(dtype=int)
    if not np.isfinite(x).all() or not np.array_equal(np.unique(y),[0,1]): parser.error('Invalid values or target')
    indices=np.arange(len(y)); trainval,test=train_test_split(indices,test_size=.2,random_state=a.seed,stratify=y)
    train_idx,val=train_test_split(trainval,test_size=.2,random_state=a.seed,stratify=y[trainval])
    scale=StandardScaler().fit(x[train_idx]); xt=scale.transform(x[train_idx]); xv=scale.transform(x[val]); xs=scale.transform(x[test])
    print(f'Data: {len(y)} records; bankrupt={y.sum()}; train/validation/test={len(train_idx)}/{len(val)}/{len(test)}')
    print('Target: 1=bankrupt; 0=not bankrupt. Fixed threshold=0.5. Confusion rows actual 0/1, columns predicted 0/1.')
    def report(name,p):
        pred=(p>=.5).astype(int); target=y[test]
        print(f'{name} | accuracy={accuracy_score(target,pred):.3f} | precision={precision_score(target,pred,zero_division=0):.3f} | recall={recall_score(target,pred,zero_division=0):.3f} | PR-AUC={average_precision_score(target,p):.3f} | ROC-AUC={roc_auc_score(target,p):.3f} | confusion={confusion_matrix(target,pred).tolist()}')
    report('Always non-bankrupt',np.zeros(len(test)))
    dt=DecisionTreeClassifier(max_depth=4,min_samples_leaf=20,class_weight='balanced',random_state=a.seed).fit(xt,y[train_idx]); report('Decision tree',dt.predict_proba(xs)[:,1])
    for name, factory in [('MLP',lambda:MLP(8)),('RBF network',lambda:RBFNetwork(8)),('ANFIS-inspired',lambda:NeuroFuzzy(8)),('Hierarchical fuzzy prototype',HierarchicalFuzzy)]:
        torch.manual_seed(a.seed); model,epoch=train(factory(),xt,y[train_idx],xv,y[val],a.seed)
        with torch.no_grad(): p=torch.sigmoid(model(torch.tensor(xs,dtype=torch.float32))).numpy()
        report(f'{name} (epoch {epoch})',p)

if __name__=='__main__': main()

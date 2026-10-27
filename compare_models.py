"""Compare thesis-inspired models; label interpretation remains provisional."""
import argparse
import copy
import warnings
import numpy as np
import pywt
import torch
from scipy.io import loadmat
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from torch import nn
from train import NeuroFuzzy


class RBFNetwork(nn.Module):
    """Learnable radial-basis hidden layer and binary output."""
    def __init__(self, features, units=12):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(units, features) * .2)
        self.log_width = nn.Parameter(torch.zeros(units))
        self.output = nn.Linear(units, 1)

    def forward(self, x):
        width = self.log_width.clamp(-3, 3).exp()
        d = ((x[:, None, :] - self.centers) ** 2).mean(dim=2)
        return self.output(torch.exp(-d / (2 * width**2))).squeeze(1)


class PairFuzzy(nn.Module):
    """Two-input fuzzy node with two Gaussian memberships per input."""
    def __init__(self):
        super().__init__()
        self.center = nn.Parameter(torch.tensor([[-.5, .5], [-.5, .5]]))
        self.log_width = nn.Parameter(torch.zeros(2, 2))
        self.coeff = nn.Parameter(torch.randn(4, 3) * .1)

    def forward(self, x):
        mu = -.5 * ((x[:, :, None] - self.center) / self.log_width.clamp(-3, 3).exp())**2
        log_w = (mu[:, 0, :, None] + mu[:, 1, None, :]).reshape(-1, 4)
        weights = torch.softmax(log_w, dim=1)
        affine = torch.cat((x, torch.ones((len(x), 1), device=x.device)), dim=1) @ self.coeff.T
        return (weights * affine).sum(dim=1)


class HierarchicalFuzzy(nn.Module):
    """8 input features -> 4 pair nodes -> 2 pair nodes -> 1 output node."""
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.ModuleList([PairFuzzy() for _ in range(n)]) for n in (4, 2, 1)])

    def forward(self, x):
        for layer in self.layers:
            x = torch.stack([node(x[:, 2*i:2*i+2]) for i, node in enumerate(layer)], dim=1)
        return x[:, 0]


def wavelet_features(X):
    # Treats the 20-feature vector as ordered signal: assumption, not yet verified.
    rows = []
    for row in X:
        packet = pywt.WaveletPacket(data=row, wavelet='haar', mode='periodization', maxlevel=4)
        rows.append(np.concatenate([node.data for node in packet.get_level(4, order='freq')]))
    return np.asarray(rows)


def genetic_selection(X, y, seed):
    """Select eight wavelet coefficients using train-only internal CV."""
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
    n, size = X.shape[1], min(8, X.shape[1])
    population = [tuple(sorted(rng.choice(n, size, replace=False))) for _ in range(12)]
    cache = {}
    def score(chromosome):
        if chromosome not in cache:
            estimator = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500))
            cache[chromosome] = cross_val_score(estimator, X[:, chromosome], y, cv=cv, scoring='balanced_accuracy').mean()
        return cache[chromosome]
    for _ in range(6):
        ranked = sorted(population, key=score, reverse=True)
        next_gen = ranked[:4]
        while len(next_gen) < 12:
            a, b = rng.choice(4, size=2, replace=True)
            pool = sorted(set(ranked[a]) | set(ranked[b]))
            child = list(rng.choice(pool, size=size, replace=False))
            if rng.random() < .5:
                pos = int(rng.integers(size))
                remaining = list(set(range(n)) - set(child))
                child[pos] = int(rng.choice(remaining))
            next_gen.append(tuple(sorted(child)))
        population = next_gen
    best = max(population, key=score)
    return list(best), score(best)


def train_torch(model, X, y, Xval, yval, seed):
    torch.manual_seed(seed)
    x = torch.tensor(X, dtype=torch.float32); target = torch.tensor(y, dtype=torch.float32)
    xv = torch.tensor(Xval, dtype=torch.float32); yv = torch.tensor(yval, dtype=torch.float32)
    opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    loss = nn.BCEWithLogitsLoss(); best_loss = float('inf'); best_epoch = 0; best_state = None
    for epoch in range(1, 501):
        model.train(); opt.zero_grad(); error = loss(model(x), target); error.backward(); opt.step()
        model.eval()
        with torch.no_grad(): value = loss(model(xv), yv).item()
        if value < best_loss - 1e-5:
            best_loss = value; best_epoch = epoch; best_state = copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 60: break
    model.load_state_dict(best_state)
    return model, best_epoch


def torch_predict(model, X):
    model.eval()
    with torch.no_grad(): return torch.sigmoid(model(torch.tensor(X, dtype=torch.float32))).numpy()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('file'); parser.add_argument('--label-row', type=int, required=True)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    torch.set_num_threads(1); torch.manual_seed(args.seed)
    m = np.asarray(loadmat(args.file)['FinanData'], dtype=np.float64)
    if m.ndim != 2 or not 0 <= args.label_row < m.shape[0] or not np.isfinite(m).all(): parser.error('Invalid matrix or label row')
    y = m[args.label_row, :].astype(int)
    if not np.array_equal(m[args.label_row, :], y) or not np.array_equal(np.unique(y), [0, 1]): parser.error('Label row must be binary with both classes')
    X = np.delete(m, args.label_row, axis=0).T
    indices = np.arange(len(y))
    trainval, test = train_test_split(indices, test_size=.2, random_state=args.seed, stratify=y)
    train, val = train_test_split(trainval, test_size=.2, random_state=args.seed, stratify=y[trainval])
    scaler = StandardScaler().fit(X[train]); Xt = scaler.transform(X[train]); Xv = scaler.transform(X[val]); Xs = scaler.transform(X[test])
    print(f'Records={len(y)}; features={X.shape[1]}; train/validation/test={len(train)}/{len(val)}/{len(test)}')
    print('PROVISIONAL: label row, 0/1 mapping, feature order and extra record need verification.\n')
    results=[]
    def report(name, prob, details=''):
        prediction=(prob >= .5).astype(int); matrix=confusion_matrix(y[test],prediction,labels=[0,1])
        acc=accuracy_score(y[test],prediction); auc=roc_auc_score(y[test],prob)
        print(f'{name}: accuracy={acc:.3f}; ROC-AUC={auc:.3f}; confusion={matrix.tolist()} {details}')
        results.append((name,acc,auc,matrix))
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', ConvergenceWarning)
        mlp=MLPClassifier(hidden_layer_sizes=(16,), max_iter=1000, random_state=args.seed, alpha=.01).fit(Xt,y[train])
    report('MLP',mlp.predict_proba(Xs)[:,1])
    tree=DecisionTreeClassifier(max_depth=4,min_samples_leaf=4,random_state=args.seed).fit(Xt,y[train])
    report('Decision tree',tree.predict_proba(Xs)[:,1])
    torch.manual_seed(args.seed); rbf,epoch=train_torch(RBFNetwork(X.shape[1]),Xt,y[train],Xv,y[val],args.seed)
    report('RBF neural network',torch_predict(rbf,Xs),f'(epoch {epoch})')
    torch.manual_seed(args.seed); anfis,epoch=train_torch(NeuroFuzzy(X.shape[1]),Xt,y[train],Xv,y[val],args.seed)
    report('ANFIS-inspired',torch_predict(anfis,Xs),f'(epoch {epoch})')
    # Selection sees train only; no validation/test labels participate.
    W=wavelet_features(X); chosen,cv_score=genetic_selection(W[train],y[train],args.seed)
    ws=StandardScaler().fit(W[train][:,chosen]); Wt=ws.transform(W[train][:,chosen]); Wv=ws.transform(W[val][:,chosen]); We=ws.transform(W[test][:,chosen])
    torch.manual_seed(args.seed); hierarchical,epoch=train_torch(HierarchicalFuzzy(),Wt,y[train],Wv,y[val],args.seed)
    report('GA + wavelet + hierarchical fuzzy prototype',torch_predict(hierarchical,We),f'(epoch {epoch}; GA internal CV balanced accuracy {cv_score:.3f}; chosen {chosen})')

if __name__ == '__main__': main()

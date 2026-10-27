"""Experimental neuro-fuzzy baseline for a MATLAB matrix with a user-selected label row."""
import argparse
import copy
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from torch import nn


class NeuroFuzzy(nn.Module):
    """Gaussian rule firing with first-order Takagi-Sugeno consequents."""
    def __init__(self, n_features, rules=8):
        super().__init__()
        self.centers = nn.Parameter(torch.randn(rules, n_features) * .2)
        self.log_scales = nn.Parameter(torch.zeros(rules, n_features))
        self.consequents = nn.Parameter(torch.randn(rules, n_features + 1) * .05)

    def forward(self, x):
        scales = self.log_scales.clamp(-3, 3).exp()
        log_firing = -0.5 * (((x[:, None, :] - self.centers) / scales) ** 2).mean(dim=2)
        weights = torch.softmax(log_firing, dim=1)
        augmented = torch.cat((x, torch.ones((len(x), 1), device=x.device)), dim=1)
        outputs = augmented @ self.consequents.T
        return (weights * outputs).sum(dim=1)


def evaluate(name, y, probability):
    pred = (probability >= 0.5).astype(int)
    print(f'\n{name}: accuracy={accuracy_score(y, pred):.3f}; ROC-AUC={roc_auc_score(y, probability):.3f}')
    print('Confusion matrix (rows actual 0/1; columns predicted 0/1):')
    print(confusion_matrix(y, pred, labels=[0, 1]))
    print(classification_report(y, pred, labels=[0, 1], target_names=['class 0', 'class 1'], zero_division=0))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('file', help='Path to .mat file')
    ap.add_argument('--variable', default='FinanData')
    ap.add_argument('--label-row', type=int, required=True, help='Zero-based index of the hypothesized label row')
    ap.add_argument('--seed', type=int, default=42)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    torch.set_num_threads(1)
    matrix = np.asarray(loadmat(args.file)[args.variable], dtype=np.float64)
    if matrix.ndim != 2 or not (0 <= args.label_row < matrix.shape[0]):
        ap.error('Expected a two-dimensional matrix and a valid label row')
    y = matrix[args.label_row, :]
    if not np.isfinite(matrix).all() or not np.array_equal(np.unique(y), [0, 1]):
        ap.error('Matrix must be finite, and label row must contain both 0 and 1')
    X = np.delete(matrix, args.label_row, axis=0).T
    y = y.astype(int)
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=args.seed, stratify=y)
    # A separate validation set chooses the stopping point; the test set remains untouched.
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=.2, random_state=args.seed, stratify=y_train)
    scaler = StandardScaler().fit(x_train)
    x_train, x_val, x_test = (scaler.transform(a) for a in (x_train, x_val, x_test))
    baseline = LogisticRegression(max_iter=2000, random_state=args.seed).fit(x_train, y_train)
    print(f'Samples: {len(y)}; features: {X.shape[1]}; train/validation/test: {len(y_train)}/{len(y_val)}/{len(y_test)}')
    print(f'PROVISIONAL: selected label row {args.label_row}; class meaning and extra record not confirmed.')
    evaluate('Logistic regression', y_test, baseline.predict_proba(x_test)[:, 1])
    Xtr = torch.tensor(x_train, dtype=torch.float32)
    Ytr = torch.tensor(y_train, dtype=torch.float32)
    Xval = torch.tensor(x_val, dtype=torch.float32)
    Yval = torch.tensor(y_val, dtype=torch.float32)
    model = NeuroFuzzy(X.shape[1]); opt = torch.optim.Adam(model.parameters(), lr=.01, weight_decay=.001)
    loss = nn.BCEWithLogitsLoss(); best = float('inf'); best_model = None; best_epoch = 0
    for epoch in range(1, 501):
        model.train(); opt.zero_grad(); error = loss(model(Xtr), Ytr); error.backward(); opt.step()
        model.eval()
        with torch.no_grad(): val_loss = loss(model(Xval), Yval).item()
        if val_loss < best - 1e-5:
            best, best_epoch, best_model = val_loss, epoch, copy.deepcopy(model.state_dict())
        if epoch - best_epoch >= 60: break
    model.load_state_dict(best_model); model.eval()
    with torch.no_grad(): probabilities = torch.sigmoid(model(torch.tensor(x_test, dtype=torch.float32))).numpy()
    print(f'Neuro-fuzzy: selected epoch {best_epoch} using validation loss {best:.3f}.')
    evaluate('Neuro-fuzzy prototype', y_test, probabilities)

if __name__ == '__main__':
    main()

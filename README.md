# Financial Bankruptcy Prediction: Neuro-Fuzzy Research

A work-in-progress modern experiment inspired by Mehran Beikahmadi's completed master's thesis in Artificial Intelligence. The thesis describes hierarchical ANFIS submodels, wavelet packet features, genetic feature selection and gradient descent for bankruptcy prediction using Tehran Stock Exchange financial data.

## Implementation status

This repository contains an **experimental PyTorch neuro-fuzzy classifier** with Gaussian rule firing and first-order Takagi–Sugeno consequents, plus a logistic regression baseline. The single neuro-fuzzy classifier is a simplified prototype. A separate five-model comparison includes a thesis-inspired hierarchical fuzzy, wavelet and genetic algorithm prototype; neither is an exact reproduction of the original thesis. The original MATLAB data file is included with the project author’s permission. Its schema and provenance still need verification.

## Provisional experiment on original file

The provided `FinanData.mat` contains a 21 × 143 matrix. The thesis describes 142 firms, 71 in each class. The last row contains 72 ones and 71 zeros, making it a plausible target row, but **its meaning and the extra record have not been confirmed**. The first 20 rows are treated as features for this exploratory run. The project author suggests that 0 may indicate bankruptcy; this class mapping is provisional and needs verification. No claims about bankrupt-firm detection can be made yet.

Using seed 42 and one stratified 91/23/29 train/validation/test split, the run produced:

| Model | Test accuracy | Test ROC-AUC | Confusion matrix (actual rows, predicted columns 0/1) |
| --- | ---: | ---: | --- |
| Logistic regression | 27/29 (93.1%) | 0.976 | [[14, 0], [2, 13]] |
| Neuro-fuzzy prototype | 27/29 (93.1%) | 0.943 | [[14, 0], [2, 13]] |

The result is **provisional**, based on only 29 test samples; feature meanings, class labels, record independence, potential target leakage, data collection timing and the extra record need verification. This split is not directly comparable to the original thesis. Model selection uses validation loss; scaling is fitted on the training split only. The score does not establish real-world performance.

## Run locally

Python 3.10+ recommended. Create and activate a virtual environment, then:

```bash
pip install -r requirements.txt
python inspect_mat.py FinanData.mat
python train.py FinanData.mat --label-row 20
```

`--label-row 20` is zero-based and is a hypothesis, not a verified schema. The script assumes columns are samples and all remaining rows are features; don't rely on its result until those assumptions are confirmed. It supports MATLAB files readable by SciPy; v7.3 requires a different reader.

## Next steps

1. Confirm the data layout, target mapping, extra record, permitted uses and any identifiers or leakage.
2. Re-evaluate with appropriate company- and time-aware splits and repeated validation.
3. Implement and test the thesis's hierarchical fuzzy structure, wavelet features and genetic selection separately.
4. Compare class-specific errors once labels are verified and document limitations.

Research and education only; not for financial decisions.

## Five-model research comparison

Run `python compare_models.py FinanData.mat --label-row 20` to train five implementations on the same split. The label row, 0/1 meaning, feature order, extra record, and any leakage remain unverified. Results below are a single exploratory run on 29 test records, not evidence of real-world predictive performance.

| Model | Test accuracy | Test ROC-AUC | Confusion matrix (actual rows, predicted columns 0/1) |
| --- | ---: | ---: | --- |
| MLP | 26/29 (89.7%) | 0.971 | [[14, 0], [3, 12]] |
| Decision tree | 20/29 (69.0%) | 0.814 | [[9, 5], [4, 11]] |
| RBF neural network | 25/29 (86.2%) | 0.943 | [[12, 2], [2, 13]] |
| ANFIS-inspired prototype | 27/29 (93.1%) | 0.943 | [[14, 0], [2, 13]] |
| GA + wavelet + hierarchical fuzzy prototype | 24/29 (82.8%) | 0.867 | [[13, 1], [4, 11]] |

The combined prototype uses a four-level Haar wavelet packet transform over each ordered feature vector, genetic selection of eight coefficients based on internal training-set cross-validation of a logistic regression proxy, and a trainable 4 → 2 → 1 hierarchy of two-input Sugeno-style fuzzy nodes. This is a **thesis-inspired approximation**, not an exact implementation or replication: the original feature ordering, genetic fitness and other training details need confirmation. The wavelet transform treats the feature order as meaningful; this assumption requires validation. Fuzzy neural models select a stopping epoch using validation loss. The MLP and decision tree use fixed settings; no model was selected using test results. Repeatable seeds do not guarantee identical results across environments.

## Real-world data: Taiwanese bankruptcy prediction

This optional experiment uses the UCI Taiwanese Bankruptcy Prediction dataset, collected from the Taiwan Economic Journal for 1999–2009. It has 6,819 records, 95 features, and 220 bankrupt records. In this dataset the label is 1 for bankrupt and 0 for non-bankrupt; **do not transfer that mapping to the thesis data**. Source: Taiwanese Bankruptcy Prediction (2020), UCI Machine Learning Repository, https://doi.org/10.24432/C5004D. License: CC BY 4.0, https://creativecommons.org/licenses/by/4.0/; attribution is required. The file is downloaded separately from UCI rather than committed to this repository.

```bash
pip install -r requirements.txt
python download_taiwan.py
python taiwan_experiment.py taiwan_bankruptcy.csv
```

The experiment uses eight predefined financial indicators, a stratified random 64/16/20 train/validation/test split and a fixed 0.5 threshold. Inputs are standardized using training data only. Neural models use positive-class weighting and validation loss for early stopping; the decision tree uses class weights. The hierarchical prototype uses eight raw financial features; it does **not** use the thesis's wavelet or genetic selection in this experiment. For the latter components see `compare_models.py`.

One run (seed 42, 1,364 test records including 44 bankrupt) produced:

| Model | Accuracy | Bankruptcy precision | Bankruptcy recall | Average precision (PR-AUC) |
| --- | ---: | ---: | ---: | ---: |
| Always predict not bankrupt | 96.8% | 0.0% | 0.0% | 0.032 |
| Decision tree | 87.1% | 17.3% | 79.5% | 0.215 |
| MLP | 86.7% | 17.2% | 81.8% | 0.388 |
| RBF network | 96.8% | 0.0% | 0.0% | 0.168 |
| ANFIS-inspired | 87.5% | 17.9% | 79.5% | 0.337 |
| Hierarchical fuzzy prototype | 89.8% | 20.9% | 77.3% | 0.343 |

Class imbalance makes accuracy misleading. For example, the MLP found 36 of 44 bankrupt cases but flagged 173 non-bankrupt records as bankrupt. The RBF model selected epoch 1 and predicted no bankrupt cases at the fixed threshold; its training and threshold need further work. Metrics vary with versions and the split. A random split can overstate real-world performance if observations from the same firm or neighboring years appear across splits, and the dataset does not establish present-day performance. Review feature timing and potential leakage, and use company/time-aware validation when appropriate metadata is available. This is not a complete reproduction of the thesis or a production financial tool.

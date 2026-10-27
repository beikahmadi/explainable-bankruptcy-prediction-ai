# Financial Bankruptcy Prediction: Neuro-Fuzzy Research

A work-in-progress project inspired by Mehran Beikahmadi's completed master's thesis in Artificial Intelligence. The research explored an evolutionary neuro-fuzzy approach to bankruptcy prediction using financial information from companies listed on the Tehran Stock Exchange.

## Research background

The thesis describes a hierarchical fuzzy inference system using ANFIS submodels, wavelet packet feature extraction, binary genetic algorithm feature selection, and gradient descent. It reports comparisons against neural-network and decision-tree methods. These are historical thesis claims; this repository does not yet reproduce or independently verify them.

## Current status

**Early exploration.** This repository currently provides a small tool to inspect MATLAB `.mat` file structure. The original MATLAB data file is not included. The modern model, training pipeline, and evaluation are not yet implemented. The original project's matrix has dimensions 21 × 143; row and column meaning, target labels, data provenance, permissions, and preprocessing still need verification before modeling.

## Inspect a MATLAB file

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python inspect_mat.py path/to/data.mat
```

The script displays variable names, dimensions and data types, not the values. Keep private or licensed data outside the repository. It supports MATLAB files that SciPy can read; v7.3 files require a different reader.

## Next steps

1. Document the dataset, labels, provenance, permitted uses and original evaluation.
2. Build a reproducible baseline with a leakage-aware train/test split.
3. Explore a modern neuro-fuzzy implementation and feature selection.
4. Compare methods using appropriate class-sensitive metrics, and document limitations.

## Responsible use

Research and education only. This project is not intended for financial decisions. Results from the original thesis cannot be assumed to generalize to current markets.

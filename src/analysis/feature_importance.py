"""Permutation feature importance (paper Phase 2.5).

Permutation importance (sklearn) under grouped CV (scenario A) and
intra-source (scenario B): importance is measured on folds where the test
source is invisible, so it is not inflated by the source structure.

Reports the RMSE loss (in orders) when each feature is permuted.

Usage:
    python -m src.analysis.feature_importance
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.model_selection import KFold, GroupKFold
from sklearn.inspection import permutation_importance

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import intra_source_folds


def perm_importance(X, y, groups, cv, n_repeat=10):
    """Permutation importance aggregated over the CV folds."""
    imp = np.zeros(len(FEATURES))
    for tr, te in cv.split(X, y, groups=groups):
        m = rf().fit(X[tr], y[tr])
        r = permutation_importance(m, X[te], y[te], n_repeats=n_repeat,
                                   random_state=0, n_jobs=1)
        imp += r.importances_mean
    return imp / cv.get_n_splits()


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()

    # Scenario A: grouped CV (new source)
    impA = perm_importance(X, y, g, GroupKFold(5))

    # Scenario B: intra-source (n >= MIN_SOURCE_SIZE)
    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    fold = intra_source_folds(big["Soil_Dataset_ID"].to_numpy())
    impB = np.zeros(len(FEATURES))
    for f in range(5):
        m = rf().fit(Xb[fold != f], yb[fold != f])
        r = permutation_importance(m, Xb[fold == f], yb[fold == f],
                                   n_repeats=10, random_state=0, n_jobs=1)
        impB += r.importances_mean
    impB /= 5

    print(f"{'feature':12s} | {'A (new source)':>20s} | {'B (intra-source)':>20s}")
    print("-" * 58)
    for i, f in enumerate(FEATURES):
        print(f"{f:12s} | {impA[i]:18.3f} | {impB[i]:18.3f}")
    print("-" * 58)
    print("RMSE loss (orders) when the feature is permuted.")
    print("Importance measured on folds with an invisible source -> not inflated.")


if __name__ == "__main__":
    main()
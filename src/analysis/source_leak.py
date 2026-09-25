"""Source-identity leak test (paper B6).

If the material features encode the source identity, then a naive CV (which
mixes sources between train and test) leaks source information into training:
the model can partly reconstruct the source offset from the features, which
inflates the naive RMSE. This script quantifies the leak by training a
classifier to predict the source from the features and reporting its accuracy
against chance, plus the R2 of the source offset regressed on the features.

Usage:
    python -m src.analysis.source_leak
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import balanced_accuracy_score

from config import FEATURES, TARGET
from src.dataset.load_data import load_sv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].fillna("?").to_numpy()

    n_src = len(np.unique(g))
    chance = 100.0 / n_src

    clf = RandomForestClassifier(300, min_samples_leaf=3, random_state=0, n_jobs=-1)
    skf = StratifiedKFold(5, shuffle=True, random_state=0)
    accs, baccs = [], []
    for tr, te in skf.split(X, g):
        clf.fit(X[tr], g[tr])
        p = clf.predict(X[te])
        accs.append(100 * float(np.mean(p == g[te])))
        baccs.append(100 * float(balanced_accuracy_score(g[te], p)))
    acc_mean = float(np.mean(accs))
    bacc_mean = float(np.mean(baccs))

    src_mean = np.array([y[g == k].mean() for k in g])
    off = src_mean - y.mean()
    lm = LinearRegression().fit(X, off)
    r2_off = 1 - ((lm.predict(X) - off) ** 2).sum() / ((off - off.mean()) ** 2).sum()

    print("=" * 70)
    print("B6) SOURCE-IDENTITY LEAK TEST")
    print("=" * 70)
    print(f"  sources: {n_src} | chance accuracy: {chance:.1f}%")
    print(f"  RF classifier source-from-features accuracy: {acc_mean:.1f}% "
          f"(balanced {bacc_mean:.1f}%)")
    print(f"  R2(features -> source offset): {r2_off:.3f}")
    print("-" * 70)
    print("Reading: if the classifier accuracy is at or below chance, the")
    print("features do NOT encode the source identity and naive CV does not")
    print("leak it through the features. The R2 of the offset regression")
    print("measures how much of the source shift is predictable from the")
    print("material (paper: ~0.02).")


if __name__ == "__main__":
    main()
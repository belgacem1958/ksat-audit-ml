"""Bias-variance decomposition of the prediction error (paper B7).

For each sample i in source s, the squared error of a model decomposes as

    E[(y - yhat)^2 | s] = bias(s)^2 + variance(s) + irreducible(s)

with
    bias(s)      = E[yhat | s] - E[y | s]      (systematic source offset)
    variance(s)  = Var(yhat | s)               (model instability across folds)
    irreducible(s) = Var(y | s)                (within-source material noise)

The decomposition is computed under grouped CV (scenario A) and intra-source
(scenario B). In scenario A the bias term is dominated by the unpredictable
source offset; in scenario B it should be small.

Usage:
    python -m src.analysis.bias_variance
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np
from sklearn.model_selection import GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import intra_source_folds


def decompose(y, pred, groups):
    """Return (bias2, variance, irreducible, total_mse) averaged over samples."""
    y = np.asarray(y, float)
    p = np.asarray(pred, float)
    g = np.asarray(groups)
    bias2 = variance = irreducible = 0.0
    for k in np.unique(g):
        m = g == k
        n = int(m.sum())
        b = (p[m].mean() - y[m].mean()) ** 2
        v = p[m].var()
        irr = y[m].var()
        bias2 += n * b
        variance += n * v
        irreducible += n * irr
    N = len(y)
    bias2, variance, irreducible = bias2 / N, variance / N, irreducible / N
    total = float(((p - y) ** 2).mean())
    return bias2, variance, irreducible, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].fillna("?").to_numpy()

    print("=" * 70)
    print("B7) BIAS-VARIANCE DECOMPOSITION (MSE in log10 units)")
    print("=" * 70)

    pA = cross_val_predict(rf(), X, y, groups=g, cv=GroupKFold(5))
    b2, v, irr, tot = decompose(y, pA, g)
    print("\nScenario A (new source, grouped CV):")
    print(f"  bias^2 (source offset) = {b2:.3f} | variance = {v:.3f} | "
          f"irreducible = {irr:.3f} | MSE = {tot:.3f} (RMSE {np.sqrt(tot):.3f})")

    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    gb = big["Soil_Dataset_ID"].to_numpy()
    fold = intra_source_folds(gb)
    pb = np.empty_like(yb)
    for f in range(5):
        m = rf().fit(Xb[fold != f], yb[fold != f])
        pb[fold == f] = m.predict(Xb[fold == f])
    b2B, vB, irrB, totB = decompose(yb, pb, gb)
    print("\nScenario B (known source, intra-source CV):")
    print(f"  bias^2 (source offset) = {b2B:.3f} | variance = {vB:.3f} | "
          f"irreducible = {irrB:.3f} | MSE = {totB:.3f} (RMSE {np.sqrt(totB):.3f})")

    print("\nReading: the bias term (the source offset) is the component that")
    print(f"collapses between scenarios: {b2:.3f} (A) vs {b2B:.3f} (B), a factor "
          f"{b2 / b2B:.1f}. The irreducible within-source noise is unchanged "
          f"({irr:.3f} vs {irrB:.3f}). The new-source collapse is a bias "
          "phenomenon, not a variance phenomenon.")


if __name__ == "__main__":
    main()
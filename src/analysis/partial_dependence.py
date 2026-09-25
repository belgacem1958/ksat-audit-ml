"""Partial dependence of the material signal (paper B9).

Partial dependence plots of the intra-source random forest (scenario B) for
the four features (log af, nf, mf, void ratio). The PDP shows the marginal
effect of each feature on log10 k_sat, averaged over the other features. It
makes the nonlinear material signal interpretable and complements the
permutation importance (which measures the effect of destroying a feature).

Usage:
    python -m src.analysis.partial_dependence
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np
from sklearn.inspection import partial_dependence

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import intra_source_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    fold = intra_source_folds(big["Soil_Dataset_ID"].to_numpy())

    m = rf().fit(Xb, yb)

    print("=" * 70)
    print("B9) PARTIAL DEPENDENCE OF THE MATERIAL SIGNAL (intra-source RF)")
    print("=" * 70)
    for i, feat in enumerate(FEATURES):
        pdp = partial_dependence(m, Xb, [i], grid_resolution=21)
        grid = pdp["grid_values"][0]
        avg = pdp["average"][0]
        lo, hi = grid[0], grid[-1]
        slope = (avg[-1] - avg[0]) / (hi - lo)
        print(f"\n{feat:12s} (range [{lo:.2f}, {hi:.2f}])")
        print(f"  PDP min {avg.min():+.3f} | max {avg.max():+.3f} | "
              f"span {avg.max() - avg.min():.3f} orders | end-to-end slope {slope:+.3f}")
        for j in range(0, 21, 5):
            print(f"    {grid[j]:8.2f} -> {avg[j]:+.3f}")

    print("\nReading: the span of each PDP is the marginal effect of the feature")
    print("on log10 k_sat. A large span = a strong material lever; a flat PDP")
    print("= a weak lever (consistent with the permutation importance).")


if __name__ == "__main__":
    main()
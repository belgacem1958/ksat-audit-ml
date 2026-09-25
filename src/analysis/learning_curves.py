"""Learning curves (paper B5).

RMSE vs training size for the three protocols (naive, grouped, intra-source).
Subsampling is done proportionally within each source so that the source
structure is preserved at every training size. Question: does the
naive/grouped gap close with more data, or is it structural?

Usage:
    python -m src.analysis.learning_curves
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import rmse, group_floor, global_floor, intra_source_folds

FRACTIONS = [0.10, 0.25, 0.50, 0.75, 1.00]


def subsample_by_source(d, frac, rng):
    """Subsample proportionally within each source (frac=1.0 keeps the order)."""
    if frac >= 1.0:
        return d
    parts = []
    for k, sub in d.groupby("Soil_Dataset_ID"):
        n = max(1, int(round(frac * len(sub))))
        parts.append(sub.sample(n, random_state=rng))
    return pd.concat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    rng = 0

    print("=" * 78)
    print("B5) LEARNING CURVES (RMSE vs training size)")
    print("=" * 78)
    print(f"{'frac':>5s} {'n':>6s} | {'naive':>7s} {'grouped':>8s} {'intra':>7s} "
          f"{'g.floor':>8s} {'o.floor':>8s}")
    print("-" * 78)

    for frac in FRACTIONS:
        sub = subsample_by_source(d, frac, rng)
        X = sub[FEATURES].to_numpy(float)
        y = sub[TARGET].to_numpy(float)
        g = sub["Soil_Dataset_ID"].fillna("?").to_numpy()

        p_naive = cross_val_predict(rf(), X, y,
                                    cv=KFold(5, shuffle=True, random_state=0))
        r_naive = rmse(y, p_naive)
        p_grp = cross_val_predict(rf(), X, y, groups=g,
                                  cv=GroupKFold(5))
        r_grp = rmse(y, p_grp)
        f_grp = group_floor(y, g)
        f_glob = global_floor(y)

        big = sub[sub["Soil_Dataset_ID"].isin(
            sub["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)]
        if len(big) > 100 and big["Soil_Dataset_ID"].nunique() >= 2:
            Xb = big[FEATURES].to_numpy(float)
            yb = big[TARGET].to_numpy(float)
            fold = intra_source_folds(big["Soil_Dataset_ID"].to_numpy())
            pb = np.empty_like(yb)
            for f in range(5):
                m = rf().fit(Xb[fold != f], yb[fold != f])
                pb[fold == f] = m.predict(Xb[fold == f])
            r_intra = rmse(yb, pb)
        else:
            r_intra = float("nan")

        print(f"{frac:5.2f} {len(sub):6d} | {r_naive:7.3f} {r_grp:8.3f} "
              f"{r_intra:7.3f} {f_grp:8.3f} {f_glob:8.3f}")

    print("-" * 78)
    print("Reading: if the naive/grouped gap persists at full size, it is")
    print("structural (the source offset is not learned by adding data).")


if __name__ == "__main__":
    main()
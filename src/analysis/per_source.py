"""Per-source analysis (paper B4).

For every source of the core: size, median k_sat, offset of the source mean
from the global mean, internal RMSE (intra-source random forest, scenario B)
and contribution to the source variance. The offset is the quantity that
scenario A cannot predict; the internal RMSE is the material signal that
scenario B can exploit.

Usage:
    python -m src.analysis.per_source
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import rmse, intra_source_folds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].fillna("?").to_numpy()

    global_mean = float(y.mean())
    global_std = float(y.std())

    print("=" * 78)
    print("B4) PER-SOURCE ANALYSIS")
    print(f"    global mean = {global_mean:.3f} (log10 m/s) | global std = {global_std:.3f}")
    print("=" * 78)
    print(f"{'source':8s} {'n':>5s} {'median k (m/s)':>14s} {'offset':>7s} "
          f"{'intra RMSE':>10s} {'var share':>9s}")
    print("-" * 78)

    big = [k for k, v in d["Soil_Dataset_ID"].value_counts().items() if v >= MIN_SOURCE_SIZE]
    intra = {}
    if big:
        idxB = d.index[d["Soil_Dataset_ID"].isin(big)].to_numpy()
        posB = np.where(d.index.isin(idxB))[0]
        XB = X[posB]
        yB = y[posB]
        gB = g[posB]
        fold = intra_source_folds(gB)
        pred = np.full(len(posB), np.nan)
        for f in range(5):
            m = rf().fit(XB[fold != f], yB[fold != f])
            pred[fold == f] = m.predict(XB[fold == f])
        for k in big:
            mask = gB == k
            intra[k] = rmse(yB[mask], pred[mask])

    total_var = float(((y - global_mean) ** 2).sum())
    for k in sorted(set(g)):
        mask = g == k
        n = int(mask.sum())
        med = float(np.median(10 ** y[mask]))
        off = float(y[mask].mean()) - global_mean
        var_share = float(((y[mask] - global_mean) ** 2).sum()) / total_var
        intra_s = f"{intra[k]:.3f}" if k in intra else "  --  "
        print(f"{k:8s} {n:5d} {med:14.2e} {off:+7.3f} {intra_s:>10s} {var_share:9.1%}")

    print("-" * 78)
    print("offset = source mean - global mean (orders); the quantity scenario A")
    print("cannot predict. intra RMSE = material signal within the source (B).")
    print("var share = contribution of the source to the total variance.")


if __name__ == "__main__":
    main()
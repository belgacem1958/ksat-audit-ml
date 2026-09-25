"""Conformal calibration across alpha levels (paper B8).

Extends conformal.py: instead of two alpha levels (80%, 90%), sweep alpha over
[0.05, 0.1, 0.2, 0.3, 0.4] and report the observed coverage in scenario B
(intra-source) and scenario A (SVSoils -> UNSODA). The result is a reliability
diagram: coverage vs nominal. Scenario B should sit near the diagonal,
scenario A below it (under-coverage).

Usage:
    python -m src.analysis.conformal_calibration
"""
import warnings
warnings.simplefilter("ignore")

import argparse

import numpy as np
from sklearn.model_selection import KFold

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv, load_unsoda
from src.models.models import rf
from src.validation.protocols import intra_source_folds

ALPHAS = [0.05, 0.10, 0.20, 0.30, 0.40]


def conformal(Xtr, ytr, Xcal, ycal, Xte, alpha):
    m = rf().fit(Xtr, ytr)
    r = np.abs(m.predict(Xcal) - ycal)
    q = np.quantile(r, 1 - alpha)
    return m.predict(Xte), q


def coverage(y, p, q):
    return 100 * float(np.mean(np.abs(p - y) <= q))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=None,
                    help="path to the real dataset_v2.csv (default: synthetic)")
    args = ap.parse_args()
    d = load_sv(args.data) if args.data else load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()

    u = load_unsoda()
    Xu = u[FEATURES].to_numpy(float)
    yu = u[TARGET].to_numpy(float)

    print("=" * 70)
    print("B8) CONFORMAL CALIBRATION (coverage vs nominal)")
    print("=" * 70)

    # Scenario B: intra-source split-conformal
    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    gb = big["Soil_Dataset_ID"].to_numpy()
    fold = intra_source_folds(gb)

    print(f"\n{'nominal':>8s} | {'B coverage':>10s} {'B width':>8s} | "
          f"{'A coverage':>10s} {'A width':>8s}")
    print("-" * 56)
    for alpha in ALPHAS:
        covs, widths = [], []
        for f in range(5):
            te = fold == f
            tr = ~te
            cal = np.zeros(len(big), bool)
            kf = KFold(5, shuffle=True, random_state=0)
            for src in np.unique(gb[tr]):
                idx = np.where((gb == src) & tr)[0]
                _, cidx = next(kf.split(idx))
                cal[idx[cidx]] = True
            fit = tr & ~cal
            p, q = conformal(Xb[fit], yb[fit], Xb[cal], yb[cal], Xb[te], alpha)
            covs.append(coverage(yb[te], p, q))
            widths.append(2 * q)
        covB = np.mean(covs)
        widB = np.mean(widths)

        # Scenario A: honest external transfer. UNSODA is SP1020 inside SVSoils,
        # so SP1020 is excluded from training; calibration on a random 20% of
        # the training set (split-conformal), test on UNSODA.
        trA = d[d["Soil_Dataset_ID"] != "SP1020"]
        XA = trA[FEATURES].to_numpy(float)
        yA = trA[TARGET].to_numpy(float)
        kf = KFold(5, shuffle=True, random_state=0)
        tr, cal = next(kf.split(XA))
        m = rf().fit(XA[tr], yA[tr])
        r_cal = np.abs(m.predict(XA[cal]) - yA[cal])
        q = np.quantile(r_cal, 1 - alpha)
        p = m.predict(Xu)
        covA = coverage(yu, p, q)
        widA = 2 * q

        print(f"{100 * (1 - alpha):7.0f}% | {covB:9.1f}% {widB:8.2f} | "
              f"{covA:9.1f}% {widA:8.2f}")

    print("-" * 56)
    print("Reading: B near the diagonal (valid), A below it (under-coverage).")
    print("The under-coverage is the source offset absent from calibration.")


if __name__ == "__main__":
    main()
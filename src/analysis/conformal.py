"""Conformal prediction (uncertainty quantification) (paper Phase 2.3).

Split-conformal with intra-source calibration (scenario B):
  - interval = prediction +/- quantile(1-alpha) of the calibration |residuals|
  - checks marginal coverage at 80% and 90%

Then applied to scenario A (new source):
  - calibration on SVSoils, test on UNSODA (external)
  - coverage must UNDER-COVER: the source offset (unpredictable) is not in
    the calibration residuals -> the interval is too narrow.

Usage:
    python -m src.analysis.conformal
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.model_selection import KFold

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv, load_unsoda
from src.models.models import rf
from src.validation.protocols import intra_source_folds


def conformal(Xtr, ytr, Xcal, ycal, Xte, alpha):
    """Split-conformal: fit on train, residuals on cal, interval on test."""
    m = rf().fit(Xtr, ytr)
    r = np.abs(m.predict(Xcal) - ycal)
    q = np.quantile(r, 1 - alpha)
    p = m.predict(Xte)
    return p, q


def coverage(y, p, q):
    return 100 * float(np.mean(np.abs(p - y) <= q))


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()

    u = load_unsoda()
    Xu = u[FEATURES].to_numpy(float)
    yu = u[TARGET].to_numpy(float)

    # ================= Scenario B: intra-source, intra-source calibration ====
    print("=" * 72)
    print("SCENARIO B - split-conformal intra-source (n>=30)")
    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    gb = big["Soil_Dataset_ID"].to_numpy()
    fold = intra_source_folds(gb)

    for alpha, lab in [(0.2, "80%"), (0.1, "90%")]:
        covs, widths = [], []
        for f in range(5):
            te = fold == f
            tr = ~te
            # calibration: 20% intra-source of the train
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
        print(f"  alpha={lab:3s}: coverage {np.mean(covs):5.1f}% "
              f"(expected {100 * (1 - alpha):.0f}%) | mean width {np.mean(widths):.2f} orders")

    # ================= Scenario A: SVSoils -> UNSODA (external) =============
    print("\n" + "=" * 72)
    print("SCENARIO A - intra-source calibration on SVSoils, test on UNSODA")
    print("(external base). The source offset is NOT in the calibration")
    print("residuals: coverage must under-cover.")
    print("NOTE: with the SYNTHETIC SVSoils the transfer to UNSODA is not")
    print("reproduced (the synthetic material signal does not match the real")
    print("UNSODA relationship), so the under-coverage of the paper is NOT")
    print("expected here. See README 'What is reproduced'.")
    foldA = intra_source_folds(g)
    m = rf().fit(X[foldA != 0], y[foldA != 0])
    r_cal = np.abs(m.predict(X[foldA == 0]) - y[foldA == 0])
    for alpha, lab in [(0.2, "80%"), (0.1, "90%")]:
        q = np.quantile(r_cal, 1 - alpha)
        p = m.predict(Xu)
        print(f"  alpha={lab:3s}: coverage {coverage(yu, p, q):5.1f}% "
              f"(expected {100 * (1 - alpha):.0f}%) | width {2 * q:.2f} orders | "
              f"RMSE {float(np.sqrt(((p - yu) ** 2).mean())):.3f}")

    # ================= Reference: RMSE vs width ==============================
    print("\n" + "=" * 72)
    print("Reading: in scenario B coverage is ~nominal; in scenario A it")
    print("under-covers because the source offset is unpredictable.")
    print("A conformal interval calibrated on training is therefore INVALID")
    print("for a new source - UQ does not replace local measurement.")


if __name__ == "__main__":
    main()
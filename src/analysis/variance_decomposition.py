"""Variance decomposition and source-effect diagnostics.

Reproduces the paper's three structural findings:

1. Intra-source transfer: the material signal is real out-of-sample. A random
   forest trained on one source and tested on another beats the global floor.
2. "Material ~ source": a material-only model under grouped CV predicts almost
   as well as the source label itself (group floor). The source offset is
   measured by regressing the per-sample offset (source mean minus global
   mean) on the material features; a low R2 means the offset is unpredictable
   from the material.
3. Hierarchical model (RF material + per-source intercept): in scenario A the
   intercept is inestimable (the model reduces to the material RF); in
   scenario B the intercept is estimated on training and the hierarchical
   model can beat the group floor.

Usage:
    python -m src.analysis.variance_decomposition
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import rmse, group_floor, global_floor, intra_source_folds


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].fillna("?").to_numpy()

    print("=" * 74)
    print(f"CORE: n={len(d)} | sources: "
          + ", ".join(f"{k}({v})" for k, v in d.Soil_Dataset_ID.value_counts().items()))
    print(f"Global floor (mean)      : {global_floor(y):.3f}")
    floor_pg = group_floor(y, g)
    print(f"Group floor (source oracle): {floor_pg:.3f}")

    # ---- 1) Intra-source transfer (out-of-sample proof of the signal) ----
    print("\n" + "=" * 74)
    print("1) OUT-OF-SAMPLE MATERIAL SIGNAL (intra-source transfers)")
    print("   RF material-only [logaf, nf, mf, e] trained on A, tested on B.")
    print("   Reference: global floor = %.3f (predict the global mean)." % y.std())
    print("   If RMSE(transfer) < global floor, the material signal TRANSFERS.")
    print(f"   {'train':8s} {'test':8s} {'n_test':>6s} {'RMSE':>6s} {'vs floor':>11s} "
          f"{'test floor':>12s}")
    big = [k for k, v in d.Soil_Dataset_ID.value_counts().items() if v >= MIN_SOURCE_SIZE]
    for tr in big:
        for te in big:
            if tr == te:
                continue
            mtr = g == tr
            mte = g == te
            m = rf().fit(X[mtr], y[mtr])
            r = rmse(y[mte], m.predict(X[mte]))
            print(f"   {tr:8s} {te:8s} {int(mte.sum()):6d} {r:6.3f} "
                  f"{r - y.std():+11.3f} {float(y[mte].mean()):12.3f}")

    # ---- 2) Material ~ source ----
    print("\n" + "=" * 74)
    print("2) ARGUMENT 'MATERIAL ~ SOURCE'")
    print("   Does a material-only model (grouped CV) predict almost as well as")
    print("   the source label itself (group floor)?")
    p_grp = cross_val_predict(rf(), X, y, groups=g, cv=GroupKFold(5))
    r_grp = rmse(y, p_grp)
    print(f"   RF material-only, grouped CV : RMSE {r_grp:.3f}")
    print(f"   Group floor (oracle)         : RMSE {floor_pg:.3f}")
    print(f"   Ratio material/source        : {r_grp / floor_pg:.3f}  "
          f"(1.0 = features carry as much info as the source)")
    src_mean = np.array([y[g == k].mean() for k in g])
    corr = float(np.corrcoef(p_grp, src_mean)[0, 1])
    print(f"   corr(material predictions, source mean) = {corr:.3f}")
    off = src_mean - y.mean()
    lm = LinearRegression().fit(X, off)
    r2_off = 1 - ((lm.predict(X) - off) ** 2).sum() / ((off - off.mean()) ** 2).sum()
    print(f"   R2(material features -> source offset) = {r2_off:.3f}")

    # ---- 3) Hierarchical model: RF material + per-source intercept ----
    print("\n" + "=" * 74)
    print("3) HIERARCHICAL MODEL: RF material + random intercept per source")
    print("   (two steps: RF on material, then offset = mean residual per source)")

    print("\n   Scenario A - NEW SOURCE (test source unknown):")
    print("   -> the random intercept is inestimable -> the model reduces to the RF.")
    pA = cross_val_predict(rf(), X, y, groups=g, cv=GroupKFold(5))
    print(f"   RF material-only (grouped CV): RMSE {rmse(y, pA):.3f}  "
          f"[group floor {floor_pg:.3f}]")
    print("   -> verdict: on an unknown source, NO model beats the group floor")
    print("     (an oracle that knows the test source mean).")

    print("\n   Scenario B - KNOWN SOURCE, NEW SOIL (intra-source 5-fold):")
    print("   -> the source intercept is estimated on training -> the hierarchical")
    print("     model can beat the group floor by adding the material info.")
    print("   Protocol (paper): each source is split into 5 folds, so that every")
    print("     fold contains soils from ALL sources (intra_source_folds).")
    big_src = [k for k, v in d.Soil_Dataset_ID.value_counts().items() if v >= MIN_SOURCE_SIZE]
    idxB = d.index[d.Soil_Dataset_ID.isin(big_src)].to_numpy()
    yB = y[d.index.isin(idxB)]
    XB = X[d.index.isin(idxB)]
    gB = g[d.index.isin(idxB)]
    posB = np.where(d.index.isin(idxB))[0]
    pos2c = {p: i for i, p in enumerate(posB)}
    print(f"   Sources kept (n>={MIN_SOURCE_SIZE}): {big_src} | n={len(idxB)}")

    def hier_predict(Xtr, ytr, gtr, Xte, gte):
        """RF material + per-source offset (0 for an unknown source)."""
        m = rf().fit(Xtr, ytr)
        base = m.predict(Xte)
        off = {}
        for k in np.unique(gtr):
            mask = gtr == k
            off[k] = float((ytr[mask] - m.predict(Xtr[mask])).mean())
        return base + np.array([off.get(k, 0.0) for k in gte])

    foldB = intra_source_folds(gB)
    src_meanB = np.array([yB[gB == k].mean() for k in gB])
    pred = {"RF material": np.full(len(idxB), np.nan),
            "hierarchical": np.full(len(idxB), np.nan),
            "group floor": src_meanB.copy(),
            "global floor": np.full(len(idxB), yB.mean())}
    for f in range(5):
        tr = foldB != f
        te = foldB == f
        m = rf().fit(XB[tr], yB[tr])
        pred["RF material"][te] = m.predict(XB[te])
        pred["hierarchical"][te] = hier_predict(XB[tr], yB[tr], gB[tr], XB[te], gB[te])
    print(f"   {'model':22s} {'RMSE':>6s} {'vs group floor':>18s}")
    for name, p in pred.items():
        r = float(np.sqrt(((p - yB) ** 2).mean()))
        print(f"   {name:22s} {r:6.3f} {r - floor_pg:+18.3f}")

    # ---- Scenario B variant: linear hierarchical (OLS + source offset) ----
    print("\n   Scenario B (variant) - LINEAR hierarchical (OLS material + source offset):")
    predL = np.full(len(idxB), np.nan)
    for f in range(5):
        tr = foldB != f
        te = foldB == f
        lm = LinearRegression().fit(XB[tr], yB[tr])
        base = lm.predict(XB[te])
        off = {}
        for kk in np.unique(gB[tr]):
            mask = gB[tr] == kk
            off[kk] = float((yB[tr][mask] - lm.predict(XB[tr][mask])).mean())
        predL[te] = base + np.array([off.get(kk, 0.0) for kk in gB[te]])
    rL = float(np.sqrt(((predL - yB) ** 2).mean()))
    print(f"   linear hierarchical: RMSE {rL:.3f}  ({rL - floor_pg:+.3f} vs group floor)")

    print(f"\n   SUMMARY scenario B (known source, new soil):")
    print(f"   global floor        : {float(yB.std()):.3f}")
    print(f"   group floor         : {float(np.sqrt(((pred['group floor'] - yB) ** 2).mean())):.3f}")
    print(f"   RF material-only    : {float(np.sqrt(((pred['RF material'] - yB) ** 2).mean())):.3f}")
    print(f"   hierarchical (RF+src): {float(np.sqrt(((pred['hierarchical'] - yB) ** 2).mean())):.3f}")
    print(f"   linear hierarchical : {rL:.3f}")


if __name__ == "__main__":
    main()
"""Model comparison under identical protocols (paper Phase 2.1).

Four model classes on the same features (log af, nf, mf, e) and the same
protocols: naive CV, grouped CV by source, leave-one-source-out (LOGO) and
intra-source (scenario B). Metrics: RMSE / bias / % within one order of
magnitude. References: global and group floors.

Usage:
    python -m src.analysis.ml_compare
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import (rmse_bias_pct, global_floor, group_floor,
                                      intra_source_folds)


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].fillna("?").to_numpy()

    floor_glob = global_floor(y)
    floor_pg = group_floor(y, g)
    print(f"n={len(d)} | global floor {floor_glob:.3f} | group floor {floor_pg:.3f}")
    print(f"{'model':12s} | {'naive':>18s} | {'grouped':>18s} | {'intra-source':>18s}")
    print("-" * 100)

    for name, m in [("RandomForest", rf(n_estimators=150)),
                    ("Linear", LinearRegression())]:
        row = []
        for label, cv in [("naive", KFold(5, shuffle=True, random_state=0)),
                          ("grouped", GroupKFold(5))]:
            p = cross_val_predict(m, X, y, groups=g, cv=cv)
            r, b, pct = rmse_bias_pct(y, p)
            row.append(f"{r:.3f}/{b:+.2f}/{pct:.0f}%")
        # Scenario B: intra-source 5-fold split (sources with n >= MIN_SOURCE_SIZE)
        big = d[d["Soil_Dataset_ID"].isin(
            d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
        Xb = big[FEATURES].to_numpy(float)
        yb = big[TARGET].to_numpy(float)
        fold = intra_source_folds(big["Soil_Dataset_ID"].to_numpy())
        pb = np.empty_like(yb)
        for f in range(5):
            m.fit(Xb[fold != f], yb[fold != f])
            pb[fold == f] = m.predict(Xb[fold == f])
        r, b, pct = rmse_bias_pct(yb, pb)
        row.append(f"{r:.3f}/{b:+.2f}/{pct:.0f}%")
        print(f"{name:12s} | " + " | ".join(row))

    print("\nFormat: RMSE/bias/% within 1 order. Intra-source = scenario B (n>=30).")


if __name__ == "__main__":
    main()
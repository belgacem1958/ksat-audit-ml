"""Bootstrap confidence intervals on the RMSEs (paper Phase 3).

Source-level bootstrap (resampling the SOURCES with replacement, not the
observations): preserves the group structure. 100 iterations.

Reports mean + 95% CI (p2.5-p97.5) for:
  - naive RMSE, grouped RMSE, global floor, group floor
  - naive/grouped gap (the ~0.3 order)
  - intra-source RMSE (scenario B) and its floor

Usage:
    python -m src.analysis.bootstrap_ci
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import rmse, group_floor, global_floor, intra_source_folds

RNG = np.random.default_rng(0)
N_BOOT = 30


def main():
    d = load_sv()
    sources = d["Soil_Dataset_ID"].unique()
    print(f"n={len(d)} | {len(sources)} sources | source bootstrap x{N_BOOT}")

    res = {k: [] for k in ["naive", "grouped", "floor_glob", "floor_pg",
                           "gap_naive_grouped", "intra", "floor_pg_intra"]}

    for it in range(N_BOOT):
        src_boot = RNG.choice(sources, size=len(sources), replace=True)
        b = pd.concat([d[d["Soil_Dataset_ID"] == s] for s in src_boot])
        X = b[FEATURES].to_numpy(float)
        y = b[TARGET].to_numpy(float)
        g = b["Soil_Dataset_ID"].to_numpy()

        res["floor_glob"].append(global_floor(y))
        res["floor_pg"].append(group_floor(y, g))

        p_naive = cross_val_predict(rf(n_estimators=100), X, y, cv=KFold(5, shuffle=True, random_state=0))
        res["naive"].append(rmse(y, p_naive))
        if len(np.unique(g)) >= 5:
            p_grp = cross_val_predict(rf(n_estimators=100), X, y, groups=g, cv=GroupKFold(5))
            res["grouped"].append(rmse(y, p_grp))
            res["gap_naive_grouped"].append(res["naive"][-1] - res["grouped"][-1])

        # Scenario B: intra-source on sources with n >= MIN_SOURCE_SIZE
        big = b[b["Soil_Dataset_ID"].isin(
            b["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)]
        if len(big) > 100 and big["Soil_Dataset_ID"].nunique() >= 2:
            Xb = big[FEATURES].to_numpy(float)
            yb = big[TARGET].to_numpy(float)
            gb = big["Soil_Dataset_ID"].to_numpy()
            fold = intra_source_folds(gb)
            pb = np.empty_like(yb)
            for f in range(5):
                m = rf(n_estimators=100).fit(Xb[fold != f], yb[fold != f])
                pb[fold == f] = m.predict(Xb[fold == f])
            res["intra"].append(rmse(yb, pb))
            res["floor_pg_intra"].append(group_floor(yb, gb))
        if (it + 1) % 25 == 0:
            print(f"  iteration {it + 1}/{N_BOOT}")

    def ci(x):
        x = np.asarray(x)
        return float(np.mean(x)), float(np.quantile(x, 0.025)), float(np.quantile(x, 0.975))

    print("\n" + "=" * 66)
    print(f"{'metric':22s} | {'mean':>8s} | {'95% CI':>18s}")
    print("-" * 66)
    for k, lab in [("naive", "naive RMSE"),
                   ("grouped", "grouped RMSE"),
                   ("gap_naive_grouped", "naive-grouped gap"),
                   ("floor_glob", "global floor"),
                   ("floor_pg", "group floor"),
                   ("intra", "intra-source RMSE (B)"),
                   ("floor_pg_intra", "group floor (B)")]:
        if res[k]:
            m, lo, hi = ci(res[k])
            print(f"{lab:22s} | {m:8.3f} | [{lo:7.3f}, {hi:7.3f}]")
    print("-" * 66)
    print("Source bootstrap: the group structure is preserved.")


if __name__ == "__main__":
    main()
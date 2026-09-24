"""Light hyperparameter tuning of the RF under grouped CV (paper Phase 2.2).

Reduced grid (n_estimators, min_samples_leaf, max_features) evaluated in
GroupKFold(5). Goal: confirm that tuning does not change the message (the
model does not beat the group floor on a new source).

Usage:
    python -m src.analysis.tuning
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, cross_val_predict
from itertools import product

from config import FEATURES, TARGET
from src.dataset.load_data import load_sv
from src.validation.protocols import group_floor


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()
    floor_pg = group_floor(y, g)

    grid = list(product([200, 400], [2, 3, 5], ["sqrt", 1.0]))
    print(f"group floor: {floor_pg:.3f}")
    print(f"{'n_est':>6s} {'mll':>4s} {'mfeat':>6s} | {'grouped RMSE':>11s} | {'vs floor':>11s}")
    print("-" * 50)
    best = None
    for ne, mll, mf in grid:
        m = RandomForestRegressor(ne, min_samples_leaf=mll, max_features=mf,
                                  random_state=0, n_jobs=-1)
        p = cross_val_predict(m, X, y, groups=g, cv=GroupKFold(5))
        r = float(np.sqrt(((p - y) ** 2).mean()))
        tag = "BEATS" if r < floor_pg else "does not beat"
        print(f"{ne:6d} {mll:4d} {str(mf):>6s} | {r:11.3f} | {tag:>11s}")
        if best is None or r < best[0]:
            best = (r, ne, mll, mf)
    print("-" * 50)
    print(f"Best: RMSE {best[0]:.3f} (n={best[1]}, mll={best[2]}, mf={best[3]})")
    print("Expected conclusion: tuning does not make the model beat the floor.")


if __name__ == "__main__":
    main()
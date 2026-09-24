"""Smoke tests for the ksat audit pipeline.

These tests run on a small generated dataset (3 sources x 60 soils) and check
that the pipeline runs end-to-end and that the structural ordering of the
paper holds: intra-source < naive < group floor < global floor < grouped.
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, KFold, cross_val_predict

from config import FEATURES, TARGET
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import (global_floor, group_floor,
                                      intra_source_folds, rmse)


def _small_dataset(tmp_path):
    """Generate a small synthetic table and return the loaded frame.

    The structural ordering (intra < naive < group floor < global floor <
    grouped) only emerges once each source has enough soils for the random
    forest to detect the source signature, so the smoke test uses the full
    7 x 260 layout with fewer trees for speed.
    """
    import subprocess
    import sys
    out = tmp_path / "dataset_v2.csv"
    subprocess.run(
        [sys.executable, "src/synthetic/generate_synthetic_dataset.py",
         "--out", str(out), "--n-sources", "7", "--n-per-source", "260",
         "--seed", "0"],
        check=True, capture_output=True)
    return load_sv(out)


def test_generator_runs_and_loads(tmp_path):
    d = _small_dataset(tmp_path)
    assert len(d) == 1820
    assert set(FEATURES).issubset(d.columns)
    assert TARGET in d.columns
    assert d[TARGET].notna().all()


def test_structural_ordering(tmp_path):
    d = _small_dataset(tmp_path)
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()

    p_naive = cross_val_predict(rf(n_estimators=100), X, y,
                                cv=KFold(5, shuffle=True, random_state=0))
    p_group = cross_val_predict(rf(n_estimators=100), X, y, groups=g,
                                cv=GroupKFold(5))
    fold = intra_source_folds(g)
    p_intra = np.empty(len(d))
    for f in range(5):
        m = rf(n_estimators=100).fit(X[fold != f], y[fold != f])
        p_intra[fold == f] = m.predict(X[fold == f])

    r_intra = rmse(y, p_intra)
    r_naive = rmse(y, p_naive)
    r_grpfl = group_floor(y, g)
    r_floor = global_floor(y)
    r_group = rmse(y, p_group)

    assert r_intra < r_naive, (r_intra, r_naive)
    assert r_naive < r_grpfl, (r_naive, r_grpfl)
    assert r_grpfl < r_floor, (r_grpfl, r_floor)
    assert r_floor < r_group, (r_floor, r_group)


def test_offset_unpredictable(tmp_path):
    d = _small_dataset(tmp_path)
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()
    from sklearn.linear_model import LinearRegression

    src_mean = np.array([y[g == k].mean() for k in g])
    off = src_mean - y.mean()
    r2 = LinearRegression().fit(X, off).score(X, off)
    assert r2 < 0.15, r2


def test_material_signal_nonlinear(tmp_path):
    d = _small_dataset(tmp_path)
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    from sklearn.linear_model import LinearRegression

    r2_lin = LinearRegression().fit(X, y).score(X, y)
    r2_rf = rf(n_estimators=100).fit(X, y).score(X, y)
    assert r2_rf > 0.5, r2_rf
    assert r2_lin < 0.2, r2_lin
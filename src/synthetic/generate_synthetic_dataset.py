#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_synthetic_dataset.py

Generate a synthetic dataset that mimics the structure of the SVSoils-derived
analysis table used in the paper (dataset_v2.csv). The SVSoils database is a
commercial product and its raw data cannot be redistributed; this generator
provides a drop-in replacement with the SAME column schema so that every
downstream script in this repository can be exercised end-to-end.

The generator reproduces the four structural facts established in the paper:
  (1) a real, strong, NON-LINEAR material signal (random forest R^2 ~ 0.8,
      linear R^2 ~ 0.05);
  (2) a small but unpredictable source offset (source R^2 ~ 0.11, linear
      predictability of the offset from material features ~ 0.02);
  (3) a naive-vs-grouped validation gap (grouped RMSE clearly above naive
      RMSE), because each source carries a NON-LINEAR feature signature that
      a random forest exploits in-sample and in naive CV but cannot use for
      an unseen source;
  (4) an intra-source RMSE below the naive RMSE (scenario B: the source is
      represented in training, the model works).

The source signature is a per-source CORRELATION between log(af) and
log(e): the sources share the same feature means (so a linear model cannot
separate them and the offset stays unpredictable) but differ in the joint
distribution of the features (so a random forest detects the source and
cannot transfer its material signal to an unseen source).

Usage:
    python src/synthetic/generate_synthetic_dataset.py \
        --out data/synthetic/dataset_v2.csv \
        --seed 0

    # 9 sources (all real SVSoils source IDs):
    python src/synthetic/generate_synthetic_dataset.py \
        --out data/synthetic/dataset_v2_9src.csv \
        --n-sources 9 --seed 0

Outputs:
    data/synthetic/dataset_v2.csv            main analysis table
    data/synthetic/predictions_stored.csv  stored-prediction columns
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RNG = np.random.default_rng


def material_signal(z1, z2, z3, z4):
    """Non-linear material signal (standardized features), log10 ksat.

    Physically motivated: ksat increases with void ratio and with the
    steepness of the retention curve (nf, mf) and decreases with the
    air-entry value (af). The signal is dominated by interaction terms, so
    that a random forest captures most of it while a linear model captures
    almost nothing (paper: RF R^2 ~ 0.79, linear R^2 ~ 0.08).
    """
    return 0.6 * (1.1 * z1 * z2 + 0.8 * z2 * z3 + 0.6 * z1 * z3
                  + 0.5 * z1 * z2 * z3 + 0.4 * z2 * z4)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="data/synthetic/dataset_v2.csv")
    ap.add_argument("--n-sources", type=int, default=7)
    ap.add_argument("--n-per-source", type=int, default=None,
                    help="soils per source (default: 260)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = RNG(args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    n_src = args.n_sources
    n_per = args.n_per_source if args.n_per_source else 260
    source_ids = ["SP1015", "SP1020", "SS1996", "US2000", "CS2000", "RS2000",
                  "SP1022", "PM6762", "SP1023"][:n_src]
    source = np.repeat(source_ids, n_per)
    n = n_src * n_per

    # ------------------------------------------------------------------
    # 1. Source signature: a per-source CORRELATION between log(af) and
    #    log(e). The sources share the same feature means, so a linear model
    #    cannot separate them (the offset stays unpredictable, paper R^2 ~
    #    0.02), but a random forest detects the source from the joint
    #    distribution (interaction splits) and cannot transfer its material
    #    signal to an unseen source (grouped CV ~ global floor).
    # ------------------------------------------------------------------
    rho = rng.normal(0.0, 1.0, size=n_src)
    rho = (rho - rho.mean()) / rho.std() * 1.3
    rho = np.clip(rho, -0.95, 0.95)
    rho_map = dict(zip(source_ids, rho))

    # Standardized features (z1 = log af, z2 = log e, z3 = nf, z4 = mf).
    z1 = rng.normal(0.0, 1.0, size=n)
    z2 = rng.normal(0.0, 1.0, size=n)
    z3 = rng.normal(0.0, 1.0, size=n)
    z4 = rng.normal(0.0, 1.0, size=n)
    for i, s in enumerate(source_ids):
        m = source == s
        r = rho_map[s]
        z2[m] = r * z1[m] + np.sqrt(1.0 - r ** 2) * z2[m]
    z1 = (z1 - z1.mean()) / z1.std()
    z2 = (z2 - z2.mean()) / z2.std()
    z3 = (z3 - z3.mean()) / z3.std()
    z4 = (z4 - z4.mean()) / z4.std()

    # Physical units (realistic SVSoils ranges).
    af = 10 ** (np.log10(10.5) + 0.6 * z1)                 # air-entry value (kPa)
    e = np.exp(np.log(0.79) + 0.35 * z2)                   # void ratio
    nf = 10 ** (np.log10(1.11) + 0.3 * z3)                 # Fredlund-Xing n
    mf = 10 ** (np.log10(1.17) + 0.3 * z4)                 # Fredlund-Xing m

    # ------------------------------------------------------------------
    # 2. Source offset, ORTHOGONALIZED against the per-source feature means.
    #    With few sources a raw normal draw can correlate with the feature
    #    signature by chance (inflating the linear predictability of the
    #    offset); orthogonalization guarantees the offset is
    #    material-unpredictable (paper: R^2 ~ 0.02). sigma ~ 0.3 log10 units.
    # ------------------------------------------------------------------
    feat_src = np.column_stack([
        np.array([z1[source == s].mean() for s in source_ids]),
        np.array([z3[source == s].mean() for s in source_ids]),
        np.array([z4[source == s].mean() for s in source_ids]),
        np.array([z2[source == s].mean() for s in source_ids]),
    ])
    offset = rng.normal(0.0, 1.0, size=n_src)
    beta = np.linalg.lstsq(feat_src, offset, rcond=None)[0]
    offset = offset - feat_src @ beta
    offset = (offset - offset.mean()) / offset.std() * 0.3
    offset_map = dict(zip(source_ids, offset))

    # ------------------------------------------------------------------
    # 3. Target: log10 ksat = material signal + source-specific interaction
    #    + source offset + noise.
    #    The source-specific interaction (a per-source slope on z1) is what
    #    makes the material signal NOT transfer to an unseen source: a random
    #    forest learns it in-sample and in naive CV but cannot use it for a
    #    new source. The slope is orthogonalized against the feature means so
    #    the additive offset stays unpredictable. It is strong enough that
    #    the grouped CV reaches the global floor, yet the intra-source RMSE
    #    (scenario B) stays below the naive RMSE because the random forest
    #    detects the source from the correlation signature.
    # ------------------------------------------------------------------
    slope = rng.normal(0.0, 1.0, size=n_src)
    beta = np.linalg.lstsq(feat_src, slope, rcond=None)[0]
    slope = slope - feat_src @ beta
    slope = (slope - slope.mean()) / slope.std() * 1.2
    slope_map = dict(zip(source_ids, slope))

    sig = material_signal(z1, z2, z3, z4)
    inter = np.array([slope_map[s] * z1[i] for i, s in enumerate(source)])
    noise = rng.normal(0.0, 0.25, size=n)
    # Shift to the real SVSoils scale: ksat median ~ 2e-5 m/s (log10 ~ -4.7).
    log10_ksat = sig + inter + np.array([offset_map[s] for s in source]) + noise - 4.7
    ksat_ms = 10 ** log10_ksat

    # ------------------------------------------------------------------
    # 4. Assemble the table with the exact schema of dataset_v2.csv.
    # ------------------------------------------------------------------
    porosity = e / (1.0 + e)
    dry_density = 2650.0 / (1.0 + e) * 1e-3                   # kg/m3 -> g/cm3 approx
    water_content = rng.uniform(0.05, 0.45, size=n)
    specific_gravity = rng.normal(2.65, 0.05, size=n)
    liquid_limit = rng.uniform(20.0, 80.0, size=n)
    plastic_limit = liquid_limit * rng.uniform(0.4, 0.7, size=n)
    d50 = 10 ** rng.normal(-1.2, 0.8, size=n)                 # mm, log-normal
    d10 = d50 * 10 ** rng.normal(-0.6, 0.3, size=n)
    d30 = d50 * 10 ** rng.normal(-0.3, 0.2, size=n)
    d60 = d50 * 10 ** rng.normal(0.3, 0.2, size=n)
    cu = d60 / np.maximum(d10, 1e-6)
    cc = d30 ** 2 / (d10 * d60)

    df = pd.DataFrame({
        "sid": [f"S{i:05d}" for i in range(n)],
        "Soil_Dataset_ID": source,
        "Plastic_Limit": plastic_limit,
        "Liquid_Limit": liquid_limit,
        "Void_Ratio": e,
        "Porosity": porosity,
        "Water_Content": water_content,
        "Dry_Density": dry_density,
        "Specific_Gravity": specific_gravity,
        "Year_Published": rng.integers(1960, 2020, size=n),
        "ksat_Test_Method": "Laboratory",
        "Laboratory_ksat": ksat_ms,
        "Field_ksat": np.nan,
        "SWCC_Test_Method": "Drying",
        "af": af,
        "nf": nf,
        "mf": mf,
        "hr": 1e4,
        "Fredlund_Error": rng.uniform(0.001, 0.05, size=n),
        "Fredlund_AEV": af * 0.8,
        "avg": af,
        "nvg": nf,
        "mvg": mf,
        "SWCC_Count": rng.integers(8, 25, size=n),
        "D10": d10,
        "D30": d30,
        "D50": d50,
        "D60": d60,
        "Cc": cc,
        "Cu": cu,
        "ksat_ms": ksat_ms,
        "log10_ksat": log10_ksat,
        "ksat_fitted": False,
        "core": True,
    })
    df.to_csv(out, index=False)

    # ------------------------------------------------------------------
    # 5. Stored-prediction columns (schema of predictions_stored.csv).
    #    These mimic the "stored predictions" audit target: a leak-prone
    #    column (Rawls_1983_ksat) is a near-perfect copy of the target.
    # ------------------------------------------------------------------
    stock = pd.DataFrame({
        "sid": df["sid"],
        "Hazens_ksat": 10 ** (log10_ksat + rng.normal(0.0, 0.5, size=n)),
        "Kozeny_Carman_ksat": 10 ** (log10_ksat + rng.normal(0.0, 0.4, size=n)),
        "Rawls_1983_ksat": ksat_ms * (1.0 + rng.normal(0.0, 0.02, size=n)),
        "Zamarin_ksat": 10 ** (log10_ksat + rng.normal(0.0, 0.6, size=n)),
    })
    stock.to_csv(out.parent / "predictions_stored.csv", index=False)

    # ------------------------------------------------------------------
    # 6. Report the structural diagnostics (should echo the paper).
    #    The offset predictability follows the paper's methodology: regress
    #    the per-sample offset (source mean minus global mean) on the
    #    material features.
    # ------------------------------------------------------------------
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.linear_model import LinearRegression

    y = df["log10_ksat"].to_numpy(float)
    X = np.column_stack([z1, z3, z4, z2])

    src_mean = np.array([y[source == s].mean() for s in source])
    r2_src = 1.0 - float(((src_mean - y) ** 2).sum()) / float(((y - y.mean()) ** 2).sum())
    r2_lin = LinearRegression().fit(X, y).score(X, y)
    r2_rf = RandomForestRegressor(300, min_samples_leaf=3,
                                  random_state=0, n_jobs=-1).fit(X, y).score(X, y)
    off = src_mean - y.mean()
    r2_off = LinearRegression().fit(X, off).score(X, off)

    print(f"Synthetic dataset written to {out.resolve()}")
    print(f"  n = {n} soils, {n_src} sources")
    print(f"  source R2        = {r2_src:.3f}   (paper: ~0.11)")
    print(f"  material linear  = {r2_lin:.3f}   (paper: ~0.08)")
    print(f"  material RF      = {r2_rf:.3f}   (paper: ~0.79)")
    print(f"  offset~material  = {r2_off:.3f}   (paper: ~0.02)")


if __name__ == "__main__":
    main()
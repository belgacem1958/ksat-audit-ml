"""Shared configuration for the ksat audit pipeline.

All paths are relative to the repository root. The synthetic dataset
(``data/synthetic/``) is a drop-in replacement for the commercial
SVSoils-derived analysis table (``dataset_v2.csv``) so that every analysis
can be reproduced end-to-end without the commercial database. The public
validation tables (UNSODA, NCHRP) are redistributed under their original
terms in ``data/public/``.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
PUBLIC_DIR = DATA_DIR / "public"
RESULTS_DIR = ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"

DATASET_V2 = SYNTHETIC_DIR / "dataset_v2.csv"
PREDICTIONS_STORED = SYNTHETIC_DIR / "predictions_stored.csv"
UNSODA_VALIDATION = PUBLIC_DIR / "unsoda_validation.csv"
NCHRP_VALIDATION = PUBLIC_DIR / "nchrp_validation.csv"

# Material features shared by every model (paper: log af, nf, mf, void ratio).
FEATURES = ["logaf", "nf", "mf", "Void_Ratio"]
TARGET = "log10_ksat"

# Reference model (paper: random forest, 300 trees, min_samples_leaf=3).
RF_PARAMS = dict(n_estimators=300, min_samples_leaf=3, random_state=0, n_jobs=-1)

# Sources with at least this many soils are used for the intra-source
# (scenario B) analyses.
MIN_SOURCE_SIZE = 30
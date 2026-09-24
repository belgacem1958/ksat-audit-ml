# Quickstart

Step-by-step reproduction of the paper's results on the synthetic dataset.

## 1. Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Requires Python ≥ 3.10. The pipeline uses only numpy, pandas and scikit-learn.

## 2. Generate the synthetic dataset

```bash
python src/synthetic/generate_synthetic_dataset.py --seed 0
```

This writes `data/synthetic/dataset_v2.csv` and
`data/synthetic/predictions_stockees.csv`, then prints the four structural
diagnostics. They should read:

```
source R2        = 0.110   (paper: ~0.11)
material linear  = 0.008   (paper: ~0.08)
material RF      = 0.786   (paper: ~0.79)
offset~material  = 0.001   (paper: ~0.02)
```

## 3. Run the analyses

Run each script from the repository root. Every script prints the paper table
it reproduces.

```bash
# Core structure: source effect, scenarios A/B, hierarchical model
python -m src.analysis.variance_decomposition

# Model comparison under the three protocols
python -m src.analysis.ml_compare

# Uncertainty quantification (conformal)
python -m src.analysis.conformal

# Bootstrap confidence interval of the naive-grouped gap
python -m src.analysis.bootstrap_ci

# Permutation feature importance
python -m src.analysis.feature_importance

# Tuning does not beat the group floor
python -m src.analysis.tuning

# ROSETTA3 comparison
python -m src.analysis.rosetta_benchmark

# Cross-base transfer matrix
python -m src.analysis.transfer_matrix

# External validation on UNSODA/NCHRP
python -m src.analysis.external_validation
```

## 4. Expected structure

The key ordering that the paper's conclusions rely on:

```
intra-source < naive < group floor < global floor < grouped
```

reproduced as (synthetic):

```
1.44 < 1.47 < 1.68 < 1.78 < 1.95
```

## 5. Tests

```bash
python -m pytest tests/ -q
```

The smoke tests check that the pipeline runs end-to-end and that the
structural ordering holds on a small generated dataset.
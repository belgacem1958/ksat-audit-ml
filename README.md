# ksat audit — multi-source machine learning for saturated hydraulic conductivity

Reproducible companion of the paper *"Variance decomposition and honest
validation of saturated hydraulic conductivity on a multi-source database"*
(Computers & Geosciences). This repository reproduces every figure and table
of the paper end-to-end: the analysis pipeline, the validation protocols, the
external validation on public databases, and the synthetic dataset that
stands in for the commercial SVSoils-derived analysis table.

## What is reproduced

The paper establishes four structural facts about the SVSoils analysis table
(`dataset_v2.csv`). Each is reproduced here on the synthetic dataset:

| Finding | Paper | This repository |
|---|---|---|
| Naive < group floor < global floor < grouped (scenario A) | 0.76 < 0.93 < 0.99 < 1.05 | 1.47 < 1.68 < 1.78 < 1.95 |
| Intra-source < naive (scenario B) | 0.71 < 0.76 | 1.44 < 1.47 |
| Source offset unpredictable from material (R²) | ~0.02 | 0.001 |
| Material signal: RF R² / linear R² | 0.79 / 0.08 | 0.786 / 0.008 |

The absolute RMSE values are scaled up relative to the paper (the synthetic
material signal is stronger than the real one) but the *ordering* of every
protocol is identical, which is what the paper's conclusions rely on.

## What is NOT reproduced (known limitations)

* **Cross-base transfer.** The synthetic SVSoils does not transfer to the real
  UNSODA/NCHRP bases the way the real SVSoils does: the synthetic material
  signal is source-specific and does not match the real physical relationship
  between the retention parameters and ksat. The transfer matrix therefore
  shows FAIL on the off-diagonal, but the *magnitude* of the transfer failure
  is not calibrated to the paper. The conformal scenario A (SVSoils → UNSODA)
  under-covers only mildly (73 % vs 80 % nominal) instead of collapsing to
  60 %.
* **Absolute RMSE scale.** The synthetic intra-source error (~1.4 orders) is
  larger than the paper's (~0.7 orders), so conformal intervals are wider.
* **Source sizes.** The synthetic dataset is balanced (7 sources × 260 soils).
  The real table is strongly unbalanced (SP1015 dominates with 1447 of 2141
  soils). The generator accepts `--n-sources 9` to mirror the real source IDs
  but not the real sizes.

## Repository layout

```
config.py                        shared paths, features, model hyperparameters
data/
  synthetic/                     generated analysis table + stored predictions
  public/                        UNSODA and NCHRP validation tables (public)
src/
  dataset/load_data.py           loaders for the three bases
  models/models.py               model factories (RF reference + zoo)
  validation/protocols.py        CV protocols and floors
  synthetic/generate_synthetic_dataset.py
  analysis/                      one script per paper result
    variance_decomposition.py    source effect, hierarchical model, scenarios A/B
    ml_compare.py                model comparison under the three protocols
    conformal.py                 split-conformal UQ, scenarios A/B
    bootstrap_ci.py              source bootstrap of the naive-grouped gap
    feature_importance.py        permutation importance (invisible-source folds)
    tuning.py                    hyperparameter tuning vs the group floor
    rosetta_benchmark.py         comparison with ROSETTA3
    transfer_matrix.py           cross-base transfer matrix
    external_validation.py       UNSODA/NCHRP validation, inverse transfer
    grouping_granularity.py      source-effect sensitivity to the grouping
    per_source.py                per-source statistics and offsets
    learning_curves.py           RMSE vs training size (naive/grouped)
    source_leak.py               source-identity leak test
    bias_variance.py             bias-variance decomposition of the collapse
    conformal_calibration.py     conformal calibration sweep (60-95 %)
    partial_dependence.py        partial dependence of the material signal
docs/
  user_guide.md                  how the pipeline works
  tutorials/quickstart.md        step-by-step reproduction
tests/
  test_pipeline.py               smoke tests
```

## Quick start

```bash
pip install -r requirements.txt

# 1. (Re)generate the synthetic analysis table
python src/synthetic/generate_synthetic_dataset.py --seed 0

# 2. Run every analysis (each prints the paper table it reproduces)
python -m src.analysis.variance_decomposition
python -m src.analysis.ml_compare
python -m src.analysis.conformal
python -m src.analysis.bootstrap_ci
python -m src.analysis.feature_importance
python -m src.analysis.tuning
python -m src.analysis.rosetta_benchmark
python -m src.analysis.transfer_matrix
python -m src.analysis.external_validation
python -m src.analysis.grouping_granularity
python -m src.analysis.per_source
python -m src.analysis.learning_curves
python -m src.analysis.source_leak
python -m src.analysis.bias_variance
python -m src.analysis.conformal_calibration
python -m src.analysis.partial_dependence

# 3. Smoke tests
python -m pytest tests/ -q
```

## Running on the real database

The seven analysis scripts added for the paper's new sections
(`grouping_granularity`, `per_source`, `learning_curves`, `source_leak`,
`bias_variance`, `conformal_calibration`, `partial_dependence`) accept
`--data <path>` to run on the real commercial analysis table instead of the
synthetic dataset:

```bash
python -m src.analysis.per_source --data /path/to/dataset_v2.csv
python -m src.analysis.learning_curves --data /path/to/dataset_v2.csv
python -m src.analysis.conformal_calibration --data /path/to/dataset_v2.csv
```

The real table is not redistributed (commercial); the synthetic dataset
reproduces the *ordering* of every protocol, which is what the paper's
conclusions rely on.

## Data

* `data/synthetic/dataset_v2.csv` — generated by
  `src/synthetic/generate_synthetic_dataset.py`. Same column schema as the
  commercial analysis table; the generator is described in
  `docs/user_guide.md`.
* `data/public/unsoda_validation.csv`, `data/public/nchrp_validation.csv` and
  `data/public/nchrp_export/` — public validation tables redistributed under
  their original terms (see `docs/user_guide.md` for provenance).

## License

MIT (see `LICENSE`). The public validation tables keep their original terms.

## Citation

If you use this repository, cite the paper (to be completed once published).
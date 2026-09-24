"""Cross-validation protocols used in the paper.

Two generalisation scenarios are distinguished:

* Scenario A — new source: the source of the soil to predict is absent from
  training. Evaluated with GroupKFold(5) on the source label and with
  leave-one-source-out (LOGO).
* Scenario B — known source, new soil: the source is represented in training.
  Evaluated with an intra-source 5-fold split (each source is split into 5
  folds so that every fold contains soils from all represented sources).

Two floors are reported everywhere. The *global floor* is the RMSE of
predicting every soil by the mean of the whole base. The *group floor* is the
RMSE of predicting every soil by the mean of its own source; it is an oracle
in scenario A but a legitimate reference in scenario B.
"""
import numpy as np
from sklearn.model_selection import KFold


def intra_source_folds(groups, n_splits=5, random_state=0):
    """Return a fold-assignment array for an intra-source split.

    Each source is split into ``n_splits`` folds, so every fold contains
    soils from all represented sources (scenario B).
    """
    g = np.asarray(groups)
    fold = np.zeros(len(g), int)
    kf = KFold(n_splits, shuffle=True, random_state=random_state)
    for src in np.unique(g):
        idx = np.where(g == src)[0]
        for f, (_, test) in enumerate(kf.split(idx)):
            fold[idx[test]] = f
    return fold


def rmse(y, p):
    """Root mean squared error in log10 units."""
    e = np.asarray(p, float) - np.asarray(y, float)
    return float(np.sqrt((e ** 2).mean()))


def rmse_bias_pct(y, p):
    """RMSE, bias and % of predictions within one order of magnitude."""
    e = np.asarray(p, float) - np.asarray(y, float)
    return (float(np.sqrt((e ** 2).mean())), float(e.mean()),
            100 * float(np.mean(np.abs(e) <= 1)))


def global_floor(y):
    """RMSE of predicting every soil by the global mean."""
    return float(np.asarray(y, float).std())


def group_floor(y, groups):
    """RMSE of predicting every soil by the mean of its own source."""
    y = np.asarray(y, float)
    g = np.asarray(groups)
    src_mean = np.array([y[g == k].mean() for k in g])
    return float(np.sqrt(((src_mean - y) ** 2).mean()))
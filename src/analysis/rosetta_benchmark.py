"""Literature benchmark: ROSETTA (paper Phase 2.4).

ROSETTA3 predicts k_sat from texture (PSD) + density. Our RF predicts k_sat
from retention parameters (af, nf, mf, e) - more informative inputs. The
comparison is CONTEXT, not a competition.

RMSE in log10(cm/day) == RMSE in log10(m/s) (constant offset cancels).

Literature values (Zhang & Schaap 2017, J. Hydrol. 547:39-53):
  ROSETTA3 PSD+BD : 0.68 log10(cm/day) on UNSODA (internal CV)
  Araya & Ghezzehei 2019 : 0.34 (PSD+BD+OC, USKSAT)
  Lilly et al. 2008 : 0.18 (HYPRES)
  Nemes et al. 2005 : 0.15 (HYPRES)

Usage:
    python -m src.analysis.rosetta_benchmark
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
from sklearn.model_selection import KFold, GroupKFold, cross_val_predict

from config import FEATURES, TARGET, MIN_SOURCE_SIZE
from src.dataset.load_data import load_sv
from src.models.models import rf
from src.validation.protocols import rmse, intra_source_folds


def main():
    d = load_sv()
    X = d[FEATURES].to_numpy(float)
    y = d[TARGET].to_numpy(float)
    g = d["Soil_Dataset_ID"].to_numpy()

    p_naive = cross_val_predict(rf(), X, y, cv=KFold(5, shuffle=True, random_state=0))
    p_grp = cross_val_predict(rf(), X, y, groups=g, cv=GroupKFold(5))
    r_naive = rmse(y, p_naive)
    r_grp = rmse(y, p_grp)

    big = d[d["Soil_Dataset_ID"].isin(
        d["Soil_Dataset_ID"].value_counts()[lambda s: s >= MIN_SOURCE_SIZE].index)].copy()
    Xb = big[FEATURES].to_numpy(float)
    yb = big[TARGET].to_numpy(float)
    fold = intra_source_folds(big["Soil_Dataset_ID"].to_numpy())
    pb = np.empty_like(yb)
    for f in range(5):
        m = rf().fit(Xb[fold != f], yb[fold != f])
        pb[fold == f] = m.predict(Xb[fold == f])
    r_intra = rmse(yb, pb)

    print("=" * 72)
    print("K_SAT BENCHMARK - RMSE in log10(cm/day) [= log10(m/s)]")
    print("=" * 72)
    print(f"{'model':38s} | {'RMSE':>6s} | {'context':>22s}")
    print("-" * 72)
    print(f"{'ROSETTA3 (PSD+BD) [Zhang&Schaap 2017]':38s} | {0.68:6.2f} | {'UNSODA, internal CV':>22s}")
    print(f"{'Araya & Ghezzehei 2019':38s} | {0.34:6.2f} | {'USKSAT, PSD+BD+OC':>22s}")
    print(f"{'Lilly et al. 2008':38s} | {0.18:6.2f} | {'HYPRES':>22s}")
    print(f"{'Nemes et al. 2005':38s} | {0.15:6.2f} | {'HYPRES':>22s}")
    print("-" * 72)
    print(f"{'RF intra-source (scenario B) [this work]':38s} | {r_intra:6.2f} | {'SVSoils, 4 features':>22s}")
    print(f"{'RF naive [this work]':38s} | {r_naive:6.2f} | {'SVSoils':>22s}")
    print(f"{'RF new source (scenario A) [this work]':38s} | {r_grp:6.2f} | {'SVSoils, source invisible':>22s}")
    print("-" * 72)
    print("Reading: intra-source our RF is at the level of ROSETTA3")
    print("BUT with more informative inputs (retention parameters).")
    print("On a new source it is clearly worse than ROSETTA3:")
    print("the inter-source generalisation cost exceeds the algorithm gain.")
    print("ROSETTA3 is NOT beaten in transfer - that is the honest point.")


if __name__ == "__main__":
    main()
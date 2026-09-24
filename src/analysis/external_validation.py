"""External validation on UNSODA and NCHRP (paper Phases 1.1-1.2).

The UNSODA and NCHRP validation tables are redistributed in ``data/public/``
(they were built by independently refitting Fredlund-Xing SWCCs from the raw
retention points; see ``docs/user_guide.md`` for the data-preparation notes).
This script runs the transfer tests:

  A) Honest scenario A: RF trained on SVSoils WITHOUT SP1020 -> UNSODA
     (UNSODA is SP1020 in SVSoils: excluding SP1020 makes the external test
     pure).
  B) Pipeline reproduction: RF trained on full SVSoils -> UNSODA.
  C) Variance decomposition on UNSODA alone, grouped by publication/site
     (is the source effect general?).
  D) Inverse transfer: UNSODA -> SVSoils (diverse->restricted asymmetry).
  E) NCHRP: RF(SVSoils) -> NCHRP (true new-source validation, NCHRP is never
     in the main SVSoils base).

Usage:
    python -m src.analysis.external_validation
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import FEATURES, TARGET
from src.dataset.load_data import load_sv, load_unsoda, load_nchrp
from src.models.models import rf
from src.validation.protocols import rmse_bias_pct, global_floor


def main():
    core = load_sv()
    u = load_unsoda()
    n = load_nchrp()

    Xc = core[FEATURES].to_numpy(float)
    yc = core[TARGET].to_numpy(float)
    Xu = u[FEATURES].to_numpy(float)
    yu = u[TARGET].to_numpy(float)
    Xn = n[FEATURES].to_numpy(float)
    yn = n[TARGET].to_numpy(float)

    print(f"SVSoils core (training): {len(core)} soils")
    print(f"UNSODA (external): {len(u)} soils | NCHRP (external): {len(n)} soils")

    # ------------------------- A) Honest scenario A -------------------------
    trA = core[core["Soil_Dataset_ID"] != "SP1020"]
    XA = trA[FEATURES].to_numpy(float)
    yA = trA[TARGET].to_numpy(float)
    m = rf().fit(XA, yA)
    pA = m.predict(Xu)
    rA, bA, pctA = rmse_bias_pct(yu, pA)
    floor_u = global_floor(yu)
    print("\n" + "=" * 70)
    print(f"A) HONEST SCENARIO A: RF(SVSoils without SP1020, n={len(trA)}) -> UNSODA (n={len(u)})")
    print(f"   RMSE {rA:.3f} | bias {bA:+.3f} | UNSODA global floor {floor_u:.3f}")
    print(f"   % within one order: {pctA:.1f}")

    # ------------------------- B) Pipeline reproduction ---------------------
    mB = rf().fit(Xc, yc)
    pB = mB.predict(Xu)
    rB, bB, _ = rmse_bias_pct(yu, pB)
    print(f"\nB) PIPELINE REPRO: RF(SVSoils full, n={len(core)}) -> UNSODA")
    print(f"   RMSE {rB:.3f} | bias {bB:+.3f}")

    # ------------------------- C) Variance on UNSODA alone ------------------
    print("\n" + "=" * 70)
    print("C) VARIANCE DECOMPOSITION ON UNSODA ALONE")
    for grp in ["publication_ID", "site_ID"]:
        g = u[grp].fillna("?").astype(str).to_numpy()
        src_mean = np.array([yu[g == k].mean() for k in g])
        r2_src = 1.0 - float(((src_mean - yu) ** 2).sum()) / float(((yu - yu.mean()) ** 2).sum())
        Xl = np.column_stack([u["logaf"], u["nf"], u["mf"], u["e"]])
        r2_lin = LinearRegression().fit(Xl, yu).score(Xl, yu)
        r2_rf = rf().fit(Xl, yu).score(Xl, yu)
        print(f"   grouped by {grp:14s}: source R2={r2_src:.3f} | material lin R2={r2_lin:.3f} "
              f"| material RF R2={r2_rf:.3f}")

    # ------------------------- D) Inverse transfer --------------------------
    print("\n" + "=" * 70)
    print("D) INVERSE TRANSFER: RF(UNSODA) -> SVSoils (asymmetry)")
    mD = rf().fit(Xu, yu)
    pD = mD.predict(Xc)
    rD, bD, _ = rmse_bias_pct(yc, pD)
    floor_s = global_floor(yc)
    print(f"   RMSE {rD:.3f} | bias {bD:+.3f} | SVSoils global floor {floor_s:.3f}")
    print(f"   (compare: RF(SVSoils)->UNSODA RMSE {rA:.3f}; does the "
          f"diverse->restricted asymmetry reproduce?)")

    # ------------------------- E) NCHRP -------------------------------------
    print("\n" + "=" * 70)
    print("E) NCHRP: RF(SVSoils) -> NCHRP (true new-source validation)")
    floor_n = global_floor(yn)
    print(f"   NCHRP global floor: {floor_n:.3f}")
    for label, mask in [("SVSoils full", np.ones(len(core), bool)),
                        ("SVSoils without SP1020", core["Soil_Dataset_ID"] != "SP1020")]:
        m = rf().fit(Xc[mask], yc[mask])
        p = m.predict(Xn)
        r, b, pct = rmse_bias_pct(yn, p)
        print(f"   RF({label}, n={int(mask.sum())}): RMSE {r:.3f} | bias {b:+.3f} | "
              f"% within 1 order {pct:.1f}")


if __name__ == "__main__":
    main()
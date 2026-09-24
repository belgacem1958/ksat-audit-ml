"""Cross-transfer matrix (paper Phase 1.3).

Three bases, shared features (log af, nf, mf, e):
  SVSoils (synthetic in this repository) | UNSODA (public) | NCHRP (public)

All train -> test transfers, RMSE in log10, compared to the global floor of
the test base. Tests the diverse->restricted asymmetry and out-of-domain
behaviour on a complete graph.

Usage:
    python -m src.analysis.transfer_matrix
"""
import warnings
warnings.simplefilter("ignore")

import numpy as np

from config import FEATURES, TARGET
from src.dataset.load_data import load_all
from src.models.models import rf
from src.validation.protocols import rmse_bias_pct


def main():
    bases = load_all()
    for k, d in bases.items():
        print(f"{k:8s}: n={len(d):5d} | k med {10 ** d[TARGET].median():.1e} m/s | "
              f"af med {d['af'].median():6.1f} | nf med {d['nf'].median():5.2f} | "
              f"mf med {d['mf'].median():5.2f} | e med {d['Void_Ratio'].median():5.2f}")

    print("\n" + "=" * 78)
    print("TRANSFER MATRIX: RMSE (bias) - row = training, column = test")
    print("=" * 78)
    hdr = "train\\test | " + " | ".join(f"{k:>8s}" for k in bases)
    print(hdr)
    print("-" * len(hdr))
    for tr_name, tr in bases.items():
        m = rf().fit(tr[FEATURES].to_numpy(float), tr[TARGET].to_numpy(float))
        cells = []
        for te_name, te in bases.items():
            p = m.predict(te[FEATURES].to_numpy(float))
            r, b, _ = rmse_bias_pct(te[TARGET].to_numpy(float), p)
            floor = float(te[TARGET].std())
            mark = "OK" if r < floor else "FAIL"
            cells.append(f"{r:6.2f} {mark}")
        print(f"{tr_name:8s} | " + " | ".join(cells))

    print("\nGlobal floors (std of the test base):")
    for k, d in bases.items():
        print(f"  {k:8s}: {float(d[TARGET].std()):.3f}")

    print("\nReading: OK = the model beats the global floor of the test base;")
    print("FAIL = it does not beat the mean (transfer failure).")
    print("The diverse->restricted asymmetry reads on the off-diagonal.")


if __name__ == "__main__":
    main()
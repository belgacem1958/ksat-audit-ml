"""Dataset loading helpers shared by all analyses.

The three tables mirror the paper's data flow:

* ``dataset_v2.csv`` — the analysis table (synthetic in this repository).
  One row per soil with measured laboratory ksat, SWCC parameters
  (af, nf, mf), void ratio and the source (``Soil_Dataset_ID``).
* ``unsoda_validation.csv`` — external validation table (UNSODA), public.
* ``nchrp_validation.csv`` — external validation table (NCHRP), public.

Every loader returns a clean table with the shared feature columns
(``logaf``, ``nf``, ``mf``, ``Void_Ratio``) and the target ``log10_ksat``.
"""
import numpy as np
import pandas as pd

from config import (DATASET_V2, UNSODA_VALIDATION, NCHRP_VALIDATION,
                    FEATURES, TARGET)


def sidfix(s):
    """Normalise soil ids read from CSV (int64/float64/object -> clean text)."""
    return (s.astype(str).str.strip().str.replace(r"\.0$", "", regex=True))


def load_sv(path=DATASET_V2):
    """Load the SVSoils analysis table (synthetic by default).

    Keeps soils with a measured (non-fitted) ksat, a fitted SWCC (af > 0)
    and a positive void ratio, then drops rows with missing features.
    """
    sv = pd.read_csv(path)
    sv["sid"] = sidfix(sv["sid"])
    sv["ksat_fitted"] = sv["ksat_fitted"].astype(bool)
    d = sv[(sv["ksat_ms"] > 0) & (~sv["ksat_fitted"])
           & (pd.to_numeric(sv["af"], errors="coerce") > 0)
           & (pd.to_numeric(sv["Void_Ratio"], errors="coerce") > 0)].copy()
    d = d.assign(logaf=np.log10(pd.to_numeric(d["af"], errors="coerce")))
    d = d.assign(**{TARGET: np.log10(d["ksat_ms"])})
    return d.dropna(subset=FEATURES + [TARGET]).copy()


def load_unsoda(path=UNSODA_VALIDATION):
    """Load the UNSODA external validation table (public)."""
    u = pd.read_csv(path)
    u["Void_Ratio"] = u["e"]
    u["logaf"] = np.log10(u["af"])
    return u.dropna(subset=FEATURES + [TARGET]).copy()


def load_nchrp(path=NCHRP_VALIDATION):
    """Load the NCHRP external validation table (public)."""
    n = pd.read_csv(path)
    n["logaf"] = np.log10(n["af"])
    return n.dropna(subset=FEATURES + [TARGET]).copy()


def load_all():
    """Return the three tables as a dict keyed by base name."""
    return {"SVSoils": load_sv(), "UNSODA": load_unsoda(), "NCHRP": load_nchrp()}
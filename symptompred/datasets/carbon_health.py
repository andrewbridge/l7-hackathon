"""Carbon Health COVID-19 Clinical Data Repository (US, Mar–Oct 2020).

Source: https://github.com/mdcollab/covidclinicaldata (CC BY-NC-SA 4.0). One CSV per weekly batch.

Mapping notes:
  * `date` is the weekly `batch_date` (the end of the week the test was in): there's no exact test date.
  * `contact` = `high_risk_interactions`: "may have been exposed through contact with a known infected person".
  * `age` is noise-added by the publisher for privacy, so `age_60` is approximate near the boundary.
  * There is no sex/gender field, so `male` is absent.
  * Symptoms were recorded at the visit, before the result was known.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import PROJECT_ROOT
from ..schema import validate

DATA_DIR = PROJECT_ROOT / "covidclinicaldata-master" / "data"
# Everything else in the data is a PCR / NAAT molecular test; this one is unspecified (possibly antigen)
AMBIGUOUS_TESTS = {"Rapid COVID-19 Test"}

SYMPTOMS = {  # raw column -> schema name
    "cough": "cough", "fever": "fever", "sore_throat": "sore_throat", "sob": "shortness_of_breath",
    "headache": "headache", "loss_of_smell": "loss_of_smell", "loss_of_taste": "loss_of_taste",
    "runny_nose": "runny_nose", "fatigue": "fatigue", "muscle_sore": "muscle_ache", "diarrhea": "diarrhoea",
}


def read_raw(path: str | Path | None = None) -> pd.DataFrame:
    files = sorted(Path(path or DATA_DIR).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {path or DATA_DIR}")
    return pd.concat([pd.read_csv(f, low_memory=False) for f in files], ignore_index=True)


def _bool(s: pd.Series) -> pd.Series:
    return s.map({True: 1, False: 0, "True": 1, "False": 0})


def load(path: str | Path | None = None, molecular_only: bool = False) -> pd.DataFrame:
    """Load as the symptompred schema. molecular_only=True drops the ambiguous 'Rapid COVID-19 Test' rows."""
    raw = read_raw(path)
    raw = raw[raw["covid19_test_results"].isin(["Positive", "Negative"])].copy()
    raw["age"] = raw["age"].where(raw["age"] >= 0)  # a few rows use -1/-2/-3 as placeholders
    if molecular_only:
        raw = raw[~raw["test_name"].isin(AMBIGUOUS_TESTS)]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["batch_date"]),
        "y": (raw["covid19_test_results"] == "Positive").astype(int),
        **{name: _bool(raw[col]) for col, name in SYMPTOMS.items()},
        "age_60": (raw["age"] >= 60).astype(float).where(raw["age"].notna()),
        "contact": _bool(raw["high_risk_interactions"]),
        "age": raw["age"].astype(float),
        "source": "carbon_health",
    })
    return validate(df.reset_index(drop=True))

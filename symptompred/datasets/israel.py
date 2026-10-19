"""Israeli Ministry of Health tested-individuals data, as used by the paper (covidpred-master/data)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd

from . import PROJECT_ROOT
from ..schema import validate

DATA_DIR = PROJECT_ROOT / "covidpred-master" / "data"
VERSIONS = {
    "006": "corona_tested_individuals_ver_006.english.csv.zip",   # May 2020: the paper's version
    "0083": "corona_tested_individuals_ver_0083.english.csv.zip",  # Nov 2020
}
PAPER_TRAIN = ("2020-03-22", "2020-03-31")
PAPER_TEST = ("2020-04-01", "2020-04-07")


def read_raw(path: Path) -> pd.DataFrame:
    """Read the CSV inside the zip, skipping macOS `__MACOSX` metadata files."""
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv") and not n.startswith("__MACOSX"))
        with z.open(name) as f:
            return pd.read_csv(f, low_memory=False)


def load(path: str | Path | None = None, version: str = "006", drop_unknown_sex: bool = False) -> pd.DataFrame:
    """Load as the symptompred schema.

    drop_unknown_sex=True reproduces the paper's filtering (and its exact row counts).
    Results of "other" (inconclusive) are always dropped.
    """
    raw = read_raw(Path(path) if path else DATA_DIR / VERSIONS[version])
    raw = raw[raw["corona_result"].isin(["positive", "negative"])]
    if drop_unknown_sex:
        raw = raw[raw["gender"].isin(["male", "female"])]
    df = pd.DataFrame({
        "date": pd.to_datetime(raw["test_date"]),
        "y": (raw["corona_result"] == "positive").astype(int),
        "cough": raw["cough"],
        "fever": raw["fever"],
        "sore_throat": raw["sore_throat"],
        "shortness_of_breath": raw["shortness_of_breath"],
        "headache": raw["head_ache"],
        "age_60": raw["age_60_and_above"].map({"Yes": 1, "No": 0}),
        "male": raw["gender"].map({"male": 1, "female": 0}),
        "contact": (raw["test_indication"] == "Contact with confirmed").astype(int),
        "source": "israel",
    })
    return validate(df.reset_index(drop=True))

"""The data convention every dataset is converted into (in the spirit of Prophet's `ds` / `y`).

One row per test, with:

| column  | required | type                 | meaning                                         |
|---------|----------|----------------------|-------------------------------------------------|
| `date`  | yes      | datetime64           | date of the test (used for time-based splits)   |
| `y`     | yes*     | int 0/1              | test result, 1 = positive (*not needed to predict) |
| features| ≥1       | float: 1, 0 or NaN   | names from `ALL_FEATURES`; NaN = not recorded   |
| `source`| no       | str                  | dataset name                                    |
| `age`   | no       | float                | age in years, where the source provides it      |
| `region`| no       | str                  | sub-national region, where available            |

A feature a dataset never collected should be **absent** (not an all-NaN column), so
`available_features` can tell "never asked" apart from "sometimes missing".
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# The paper's eight features, under neutral names
CORE_FEATURES = (
    "cough", "fever", "sore_throat", "shortness_of_breath", "headache",
    "age_60", "male", "contact",
)
# Collected by some later datasets but not the Israeli one
EXTRA_FEATURES = (
    "loss_of_smell", "loss_of_taste", "runny_nose", "fatigue", "muscle_ache", "diarrhoea",
)
ALL_FEATURES = CORE_FEATURES + EXTRA_FEATURES
# Facts about the person rather than reports; never assumed absent when missing
DEMOGRAPHIC_FEATURES = ("age_60", "male")
META_COLUMNS = ("source", "age", "region")


class SchemaError(ValueError):
    """Raised when a DataFrame doesn't follow the symptompred convention."""


def validate(df: pd.DataFrame, require_y: bool = True) -> pd.DataFrame:
    """Check a DataFrame follows the convention and return a cleaned copy (dtypes normalised)."""
    missing = [c for c in ("date", "y") if c not in df.columns and (c == "date" or require_y)]
    if missing:
        raise SchemaError(f"Missing required column(s): {missing}. Expected 'date' and 'y'.")

    features = [c for c in df.columns if c in ALL_FEATURES]
    if not features:
        raise SchemaError(f"No feature columns found. Use names from symptompred.ALL_FEATURES: {ALL_FEATURES}")

    unknown = [c for c in df.columns if c not in ALL_FEATURES + META_COLUMNS + ("date", "y")]
    if unknown:
        raise SchemaError(f"Unrecognised column(s): {unknown}. Rename to a known feature or drop them.")

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"])
    if out["date"].isna().any():
        raise SchemaError(f"'date' has {out['date'].isna().sum()} missing values; drop or fill them first.")

    if "y" in out.columns:
        if out["y"].isna().any() or not out["y"].isin([0, 1]).all():
            raise SchemaError("'y' must be 0/1 with no missing values.")
        out["y"] = out["y"].astype(int)

    for f in features:
        values = pd.to_numeric(out[f], errors="raise").astype(float)
        bad = ~values.isna() & ~values.isin([0.0, 1.0])
        if bad.any():
            raise SchemaError(f"Feature '{f}' must contain only 0, 1 or NaN (found e.g. {values[bad].iloc[0]}).")
        out[f] = values
    return out


def available_features(df: pd.DataFrame, candidates=ALL_FEATURES) -> list[str]:
    """Features present in `df` with at least one non-missing value, in canonical order."""
    return [f for f in candidates if f in df.columns and df[f].notna().any()]


def common_features(*dfs: pd.DataFrame, candidates=ALL_FEATURES) -> list[str]:
    """Features available in every one of `dfs`, in canonical order."""
    sets = [set(available_features(d, candidates)) for d in dfs]
    return [f for f in candidates if all(f in s for s in sets)]


def pool(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Stack several schema DataFrames into one, labelling each row's `source` with its dict key.

    Features a dataset never collected are NaN for its rows. SymptomModel(missing="absent") knows
    the difference between those and genuine blanks because it fills per source.
    """
    return pd.concat([validate(d).assign(source=name) for name, d in datasets.items()], ignore_index=True)


def feature_summary(datasets: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Coverage table: share of rows with each feature recorded, per dataset (blank = never collected)."""
    rows = {}
    for name, d in datasets.items():
        rows[name] = {f: d[f].notna().mean() if f in d.columns else np.nan for f in ALL_FEATURES}
    return pd.DataFrame(rows)

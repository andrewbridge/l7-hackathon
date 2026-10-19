"""Shared fixtures: small, deterministic synthetic datasets in the symptompred schema."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pandas as pd
import pytest

from symptompred.datasets import PROJECT_ROOT


def make_country(name: str, n: int, seed: int, start: str, extra: bool = False, with_sex: bool = True,
                 nan_rate: float = 0.0) -> pd.DataFrame:
    """Synthetic schema data with a known logistic structure (contact and fever matter most)."""
    rng = np.random.default_rng(seed)
    cols = {f: rng.binomial(1, p, n).astype(float) for f, p in
            [("cough", 0.2), ("fever", 0.15), ("sore_throat", 0.1), ("shortness_of_breath", 0.05),
             ("headache", 0.1), ("age_60", 0.2), ("contact", 0.15)]}
    if with_sex:
        cols["male"] = rng.binomial(1, 0.5, n).astype(float)
    if extra:
        cols["loss_of_smell"] = rng.binomial(1, 0.08, n).astype(float)
    logit = (-3 + 2.5 * cols["contact"] + 1.5 * cols["fever"] + 0.8 * cols["cough"] + 0.5 * cols["age_60"]
             + (1.5 * cols["loss_of_smell"] if extra else 0))
    y = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    df = pd.DataFrame({"date": pd.Timestamp(start) + pd.to_timedelta(rng.integers(0, 30, n), unit="D"),
                       "y": y, **cols, "source": name})
    if nan_rate:
        for f in ("fever", "contact"):
            df.loc[rng.random(n) < nan_rate, f] = np.nan
    return df


@pytest.fixture
def country_a() -> pd.DataFrame:
    """Core features only (like Israel); no missing values."""
    return make_country("a", 3000, 1, "2020-03-01")


@pytest.fixture
def country_b() -> pd.DataFrame:
    """No sex field, an extra symptom, and some blank fever / contact values (like Carbon Health)."""
    return make_country("b", 3000, 2, "2020-06-01", extra=True, with_sex=False, nan_rate=0.1)


@pytest.fixture
def two_countries(country_a, country_b) -> dict[str, pd.DataFrame]:
    return {"a": country_a, "b": country_b}


def _real_data_available() -> bool:
    return (PROJECT_ROOT / "covidpred-master" / "data").exists()


requires_data = pytest.mark.skipif(not _real_data_available(), reason="real datasets not present")

import numpy as np
import pandas as pd
import pytest

import symptompred as sp


def test_validate_normalises_types(country_a):
    df = country_a.assign(date=country_a["date"].astype(str), cough=country_a["cough"].astype(int))
    out = sp.validate(df)
    assert out["date"].dtype.kind == "M"
    assert out["cough"].dtype == float
    assert out["y"].dtype == int


@pytest.mark.parametrize("change, message", [
    (lambda d: d.drop(columns="date"), "Missing required"),
    (lambda d: d.drop(columns="y"), "Missing required"),
    (lambda d: d.assign(mystery=1), "Unrecognised"),
    (lambda d: d.assign(cough=2.0), "only 0, 1 or NaN"),
    (lambda d: d.assign(y=np.nan), "'y' must be 0/1"),
    (lambda d: d[["date", "y"]], "No feature columns"),
])
def test_validate_errors(country_a, change, message):
    with pytest.raises(sp.SchemaError, match=message):
        sp.validate(change(country_a))


def test_validate_prediction_data_needs_no_y(country_a):
    assert "y" not in sp.validate(country_a.drop(columns="y"), require_y=False)


def test_available_and_common_features(country_a, country_b):
    assert "male" in sp.available_features(country_a)
    assert "male" not in sp.available_features(country_b)
    assert "loss_of_smell" in sp.available_features(country_b)
    # An all-NaN column counts as "not available"
    assert "headache" not in sp.available_features(country_a.assign(headache=np.nan))
    common = sp.common_features(country_a, country_b)
    assert "male" not in common and "loss_of_smell" not in common and "contact" in common
    assert common == [f for f in sp.ALL_FEATURES if f in common]  # canonical order


def test_pool_labels_sources_and_unions_columns(country_a, country_b):
    pooled = sp.pool({"first": country_a, "second": country_b})
    assert len(pooled) == len(country_a) + len(country_b)
    assert pooled["source"].value_counts().to_dict() == {"first": 3000, "second": 3000}
    # Features one country never collected are NaN for its rows
    assert pooled.loc[pooled["source"] == "first", "loss_of_smell"].isna().all()
    assert pooled.loc[pooled["source"] == "second", "male"].isna().all()


def test_feature_summary(two_countries):
    summary = sp.feature_summary(two_countries)
    assert summary.loc["cough", "a"] == 1.0
    assert np.isnan(summary.loc["male", "b"])
    assert summary.loc["fever", "b"] == pytest.approx(0.9, abs=0.03)

import numpy as np
import pandas as pd
import pytest

import symptompred as sp


def test_fit_predict_is_deterministic(country_a):
    p1 = sp.SymptomModel(seed=3).fit(country_a).predict(country_a)
    p2 = sp.SymptomModel(seed=3).fit(country_a).predict(country_a)
    np.testing.assert_array_equal(p1, p2)
    assert ((p1 > 0) & (p1 < 1)).all()


def test_learns_the_known_structure(country_a):
    train, test = sp.temporal_split(country_a)
    result = sp.SymptomModel().fit(train).evaluate(test, bootstrap=0)
    assert result["auroc"] > 0.75


def test_feature_resolution(country_a, country_b):
    assert sp.SymptomModel().fit(country_a).features_ == list(sp.CORE_FEATURES)
    assert "loss_of_smell" not in sp.SymptomModel(features="core").fit(country_b).features_
    assert "loss_of_smell" in sp.SymptomModel(features="all").fit(country_b).features_
    with pytest.raises(ValueError, match="not available"):
        sp.SymptomModel(features=["male"]).fit(country_b)


def test_predict_raw_is_log_odds(country_a):
    m = sp.SymptomModel().fit(country_a)
    np.testing.assert_allclose(m.predict(country_a), 1 / (1 + np.exp(-m.predict_raw(country_a))))


def test_missing_absent_fills_only_collected_features_per_source(country_a, country_b):
    pooled = sp.pool({"a": country_a, "b": country_b})
    m = sp.SymptomModel(features="all", missing="absent")
    m.features_ = sp.available_features(pooled)
    filled = m._handle_missing(pooled)
    a, b = filled["source"] == "a", filled["source"] == "b"
    # Country b collected fever/contact but left some blank: those become 0 ...
    assert filled.loc[b, ["fever", "contact"]].notna().all().all()
    # ... but a feature a country never collected stays NaN
    assert filled.loc[a, "loss_of_smell"].isna().all()
    # Demographics are never filled
    assert filled.loc[b, "male"].isna().all()


def test_missing_absent_never_fills_demographics():
    df = pd.DataFrame({"date": ["2020-01-01"] * 4, "y": [0, 1, 0, 1], "cough": [1, np.nan, 0, 1],
                       "age_60": [1, np.nan, 0, 0], "source": "x"})
    m = sp.SymptomModel(missing="absent")
    m.features_ = ["cough", "age_60"]
    filled = m._handle_missing(sp.validate(df))
    assert filled["cough"].tolist() == [1, 0, 0, 1]
    assert np.isnan(filled["age_60"].iloc[1])


def test_native_missing_leaves_nans(country_b):
    m = sp.SymptomModel(missing="native")
    m.features_ = ["fever"]
    assert m._handle_missing(country_b)["fever"].isna().sum() == country_b["fever"].isna().sum()


def test_invalid_missing_option():
    with pytest.raises(ValueError):
        sp.SymptomModel(missing="zero")


def test_warns_when_a_feature_is_absent_at_prediction(country_a, country_b):
    m = sp.SymptomModel().fit(country_a)
    with pytest.warns(UserWarning, match="not in this data"):
        p = m.predict(country_b)
    assert len(p) == len(country_b)


def test_balance_sources_gives_each_source_equal_total_weight(country_a, country_b, monkeypatch):
    seen = {}
    import lightgbm as lgb
    real_dataset = lgb.Dataset

    def spy(data, label=None, weight=None, **kw):
        seen.setdefault("weights", []).append((data.index, weight))
        return real_dataset(data, label, weight=weight, **kw)

    monkeypatch.setattr(lgb, "Dataset", spy)
    pooled = sp.pool({"a": country_a, "b": country_b.iloc[:1000]})
    sp.SymptomModel(features="all", num_boost_round=5).fit(pooled, sample_weight="balance_sources")
    idx_tr, w_tr = seen["weights"][0]
    idx_va, w_va = seen["weights"][1]
    w = pd.concat([pd.Series(np.asarray(w_tr), index=idx_tr), pd.Series(np.asarray(w_va), index=idx_va)])
    totals = w.groupby(pooled.loc[w.index, "source"]).sum()
    assert totals["a"] == pytest.approx(totals["b"])


def test_balance_sources_needs_source(country_a):
    with pytest.raises(ValueError, match="source"):
        sp.SymptomModel().fit(country_a.drop(columns="source"), sample_weight="balance_sources")


def test_ensemble_and_trees(country_a):
    m = sp.SymptomModel(n_models=3).fit(country_a)
    assert len(m.n_trees_) == 3 and all(n > 0 for n in m.n_trees_)
    assert sp.SymptomModel.paper().early_stopping_rounds == 5


def test_evaluate_and_shap(country_a):
    m = sp.SymptomModel().fit(country_a)
    out = m.evaluate(country_a, bootstrap=20, sensitivity=0.8)
    assert {"auroc", "auprc", "auroc_lo", "auroc_hi", "wp_specificity"} <= set(out.index)
    expl = m.shap_values(country_a.iloc[:200])
    assert expl.values.shape == (200, len(m.features_))
    # SHAP values add up to the log-odds
    np.testing.assert_allclose(expl.values.sum(1) + expl.base_values, m.predict_raw(country_a.iloc[:200]), atol=1e-6)


def test_unfitted_model_raises(country_a):
    with pytest.raises(RuntimeError):
        sp.SymptomModel().predict(country_a)

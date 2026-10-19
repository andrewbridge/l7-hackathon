import warnings

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier

import symptompred as sp
from symptompred.evaluate import _auc_ap


# ------------------------------------------------------------------ EstimatorModel
def test_estimator_model_indicators_for_features_with_missing_values(country_b):
    m = sp.EstimatorModel(LogisticRegression(max_iter=1000), features="all").fit(country_b)
    assert m.indicator_features_ == ["fever", "contact"]
    assert m.columns_[-2:] == ["fever_missing", "contact_missing"]
    X = m._prepare(m._X(country_b))
    assert not X.isna().any().any()
    np.testing.assert_array_equal(X["fever_missing"], country_b["fever"].isna())


def test_estimator_model_native_nan_passthrough(country_b):
    m = sp.EstimatorModel(HistGradientBoostingClassifier(max_iter=20), features="all",
                          missing_indicators=False).fit(country_b)
    assert m.indicator_features_ == [] and "fever_missing" not in m.columns_
    assert m._prepare(m._X(country_b))["fever"].isna().sum() == country_b["fever"].isna().sum()


def test_estimator_model_missing_absent_per_source(country_a, country_b):
    pooled = sp.pool({"a": country_a, "b": country_b})
    m = sp.EstimatorModel(LogisticRegression(max_iter=1000), features="all", missing="absent").fit(pooled)
    # Blank fever/contact in b are filled as "not reported"; loss_of_smell (never collected by a) and
    # male (demographic, never collected by b) still need indicators
    assert m.indicator_features_ == ["male", "loss_of_smell"]


def test_estimator_model_predicts_like_the_raw_estimator(country_a):
    """Predicting on distinct rows and mapping back must give exactly the estimator's own predictions."""
    m = sp.EstimatorModel(LogisticRegression(max_iter=1000)).fit(country_a)
    direct = m.estimators_[0].predict_proba(m._prepare(m._X(country_a)))[:, 1]
    np.testing.assert_allclose(m.predict(country_a), direct)
    np.testing.assert_allclose(m.predict(country_a), 1 / (1 + np.exp(-m.predict_raw(country_a))))


def test_estimator_model_is_seeded_and_ensembles(country_a):
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=10, min_samples_leaf=20)
    p1 = sp.EstimatorModel(rf, seed=1).fit(country_a).predict(country_a)
    p2 = sp.EstimatorModel(rf, seed=1).fit(country_a).predict(country_a)
    np.testing.assert_array_equal(p1, p2)
    assert len(sp.EstimatorModel(rf, n_models=3).fit(country_a).estimators_) == 3


def test_estimator_model_sample_weights(country_a):
    w = np.where(country_a["y"] == 1, 5.0, 1.0)
    p_plain = sp.EstimatorModel(LogisticRegression(max_iter=1000)).fit(country_a).predict(country_a)
    p_weighted = sp.EstimatorModel(LogisticRegression(max_iter=1000)).fit(country_a, sample_weight=w).predict(country_a)
    assert p_weighted.mean() > p_plain.mean()
    with pytest.warns(UserWarning, match="doesn't support sample weights"):
        sp.EstimatorModel(KNeighborsClassifier(5)).fit(country_a, sample_weight=w)


def test_estimator_model_shap_is_additive_and_folds_indicators(country_b):
    m = sp.EstimatorModel(LogisticRegression(max_iter=1000), features="all").fit(country_b)
    rows = country_b.iloc[:300]
    expl = m.shap_values(rows)
    assert expl.feature_names == m.features_  # indicator columns folded into their features
    np.testing.assert_allclose(expl.values.sum(1) + expl.base_values, m.predict_raw(rows), atol=1e-6)
    assert np.isnan(expl.data[:, m.features_.index("fever")]).sum() == rows["fever"].isna().sum()


# ------------------------------------------------------------------ source_feature
def test_source_feature_columns(country_a, country_b):
    pooled = sp.pool({"a": country_a, "b": country_b})
    m = sp.SymptomModel(features="all", source_feature=True, num_boost_round=20).fit(pooled)
    assert m.sources_ == ["a", "b"]
    X = m._X(pooled)
    assert {"source_a", "source_b"} <= set(X.columns)
    assert (X["source_a"] == (pooled["source"] == "a")).all()
    # An unseen country gets all zeros
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        unseen = m._X(country_a.assign(source="c"))
        assert len(m.predict(country_a.assign(source="c"))) == len(country_a)
    assert (unseen[["source_a", "source_b"]] == 0).all().all()


def test_source_feature_with_estimator_model(country_a, country_b):
    pooled = sp.pool({"a": country_a, "b": country_b})
    m = sp.EstimatorModel(LogisticRegression(max_iter=1000), features="all", source_feature=True).fit(pooled)
    assert "source_b" in m.columns_


def test_source_feature_needs_source(country_a):
    with pytest.raises(ValueError, match="source"):
        sp.SymptomModel(source_feature=True).fit(country_a.drop(columns="source"))


def test_recency_weights():
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01", "2020-01-08", "2020-01-15"])})
    np.testing.assert_allclose(sp.recency_weights(df, half_life_days=7), [0.25, 0.5, 1.0])


# ------------------------------------------------------------------ CellRateClassifier
def _cells():
    rng = np.random.default_rng(0)
    X = rng.binomial(1, 0.4, (4000, 3)).astype(float)
    y = rng.binomial(1, 0.1 + 0.2 * X[:, 0] + 0.3 * X[:, 0] * X[:, 1])
    return X, y


def test_cell_rate_alpha_zero_is_raw_rates():
    X, y = _cells()
    p = sp.CellRateClassifier(alpha=0).fit(X, y).predict_proba(X)[:, 1]
    raw = pd.DataFrame(X).assign(y=y).groupby([0, 1, 2])["y"].transform("mean").values
    np.testing.assert_allclose(p, raw)


def test_cell_rate_large_alpha_is_backoff():
    X, y = _cells()
    clf = sp.CellRateClassifier(alpha=1e9).fit(X, y)
    np.testing.assert_allclose(clf.predict_proba(X)[:, 1], clf.backoff_.predict_proba(X)[:, 1], atol=1e-5)


def test_cell_rate_unseen_cell_uses_backoff():
    X, y = _cells()
    clf = sp.CellRateClassifier(alpha=5).fit(X[X[:, 2] == 0], y[X[:, 2] == 0])
    new = np.array([[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(clf.predict_proba(new), clf.backoff_.predict_proba(new))


def test_cell_rate_captures_interactions_logistic_misses():
    X, y = _cells()
    cell = roc_auc_score(y, sp.CellRateClassifier(alpha=1).fit(X, y).predict_proba(X)[:, 1])
    lr = roc_auc_score(y, LogisticRegression().fit(X, y).predict_proba(X)[:, 1])
    assert cell >= lr


# ------------------------------------------------------------------ ScorecardClassifier
def test_scorecard_integer_points_and_ranking(country_a):
    m = sp.EstimatorModel(sp.ScorecardClassifier(max_points=10)).fit(country_a)
    points = m.estimators_[0].points_
    assert points.dtype.kind == "i" and points.abs().max() == 10
    assert points.idxmax() == "contact"
    lr = sp.EstimatorModel(LogisticRegression(max_iter=1000)).fit(country_a)
    assert spearmanr(m.predict(country_a), lr.predict(country_a)).statistic > 0.95


# ------------------------------------------------------------------ registry and ensembles
@pytest.mark.parametrize("name", list(sp.registry()))
def test_every_registered_model_fits_and_predicts(name, country_a, country_b):
    pooled = sp.pool({"a": country_a.iloc[:1500], "b": country_b.iloc[:1500]})
    train, test = sp.temporal_split(pooled)
    model = sp.registry()[name](features="all", missing="absent", source_feature=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p = model.fit(train).predict(test)
    assert p.shape == (len(test),) and np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert roc_auc_score(test["y"], p) > 0.6
    assert name in sp.FAMILIES


def test_registry_params_override():
    m = sp.registry()["XGBoost"](params={"max_depth": 2})
    assert m.estimator.get_params()["max_depth"] == 2
    assert sp.registry()["LightGBM (tuned params)"]().params["num_leaves"] == 7


@pytest.mark.parametrize("method", ["average", "stack"])
def test_ensemble(method, country_a):
    models = sp.registry()
    members = {k: models[k] for k in ["Logistic regression", "Naive Bayes"]}
    train, test = sp.temporal_split(country_a)
    ens = sp.EnsembleModel(members, method=method).fit(train)
    p = ens.predict(test)
    assert ens.evaluate(test, bootstrap=0)["auroc"] > 0.75
    if method == "average":
        expected = np.mean([m.predict_raw(test) for m in ens.models_.values()], axis=0)
        np.testing.assert_allclose(ens.predict_raw(test), expected)
    assert len(p) == len(test)

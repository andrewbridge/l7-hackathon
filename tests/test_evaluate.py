import numpy as np
import pandas as pd
import pytest

import symptompred as sp
from symptompred.evaluate import Predictions


def test_metrics_hand_computed():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    m = sp.metrics(y, p)
    assert m["auroc"] == pytest.approx(0.75)
    assert m["auprc"] == pytest.approx((1 + 2 / 3) / 2)
    assert m["brier"] == pytest.approx(np.mean((p - y) ** 2))
    assert m["positives"] == 2 and m["prevalence"] == 0.5


def test_metrics_single_class_is_nan():
    m = sp.metrics(np.zeros(5), np.linspace(0, 1, 5))
    assert np.isnan(m["auroc"]) and np.isnan(m["auprc"])


def test_bootstrap_ci_brackets_the_estimate():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.3, 2000)
    p = y * 0.3 + rng.random(2000)
    ci = sp.bootstrap_ci(y, p, n=200)
    auc = sp.metrics(y, p)["auroc"]
    assert ci["auroc_lo"] < auc < ci["auroc_hi"]
    assert ci == sp.bootstrap_ci(y, p, n=200)  # seeded


def test_working_point_hand_computed():
    y = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    p = np.array([0.9, 0.8, 0.7, 0.2, 0.6, 0.3, 0.1, 0.05])
    wp = sp.working_point(y, p, sensitivity=0.75)
    assert wp["threshold"] == 0.7
    # Flags 0.9, 0.8, 0.7: three of four positives, no negatives
    assert wp["sensitivity"] == 0.75 and wp["specificity"] == 1.0
    assert wp["ppv"] == 1.0 and wp["npv"] == 0.8 and wp["flagged"] == 3 / 8
    assert sp.working_point(y, p, sensitivity=1.0)["threshold"] == 0.2


def test_temporal_split_has_no_date_overlap(country_a):
    train, test = sp.temporal_split(country_a, test_fraction=0.3)
    assert train["date"].max() < test["date"].min()
    assert len(train) + len(test) == len(country_a)


def test_date_split_inclusive_windows(country_a):
    train, test = sp.date_split(country_a, ("2020-03-01", "2020-03-15"), ("2020-03-16", "2020-03-30"))
    assert train["date"].max() <= pd.Timestamp("2020-03-15") < test["date"].min()
    assert train["date"].min() == pd.Timestamp("2020-03-01")


def test_compare_shapes_and_like_for_like_rows(two_countries):
    res = sp.compare(two_countries, unseen_on="test")
    assert len(res) == 4
    assert set(res["kind"]) == {"within", "unseen"}
    preds = res.attrs["predictions"]
    # With unseen_on="test", every model scored on b sees exactly b's test rows
    np.testing.assert_array_equal(preds[("a", "b")][0], preds[("b", "b")][0])
    assert "male" not in res.attrs["features"]  # common features only
    assert sp.pivot(res).shape == (2, 2)


def test_compare_full_uses_all_rows_of_other_dataset(two_countries):
    res = sp.compare(two_countries, unseen_on="full")
    assert res.set_index(["train", "test"]).loc[("a", "b"), "n"] == 3000
    with pytest.raises(ValueError):
        sp.compare(two_countries, unseen_on="other")


def test_compare_holdout_is_never_trained_on(two_countries, country_a):
    res = sp.compare({"a": two_countries["a"]}, holdout={"b": two_countries["b"]})
    assert list(res["train"]) == ["a", "a"]
    assert list(res["test"]) == ["a", "b"]


def test_results_with_predictions_concatenate(two_countries):
    r1 = sp.compare(two_countries)
    r2 = sp.compare(two_countries, features=["cough", "fever"])
    combined = pd.concat([r1, r2])  # used to raise "truth value of an array is ambiguous"
    assert len(combined) == 8
    assert isinstance(r1.attrs["predictions"], Predictions)


def test_leave_one_out(two_countries):
    res = sp.leave_one_out(two_countries, features="common")
    assert list(res["held_out"]) == ["a", "b"]
    assert list(res["trained_on"]) == ["b", "a"]
    assert set(res.attrs["predictions"]) == {"a", "b"}


def test_forward_selection_never_uses_test_labels(country_a):
    train, test = sp.temporal_split(country_a)
    shuffled = test.assign(y=np.random.default_rng(0).permutation(test["y"].values))
    kw = dict(candidates=["cough", "fever", "contact", "age_60"], num_boost_round=50)
    a = sp.forward_selection(train, test, **kw)
    b = sp.forward_selection(train, shuffled, **kw)
    assert list(a["added"]) == list(b["added"])
    assert a["added"].iloc[0] == "contact"  # the strongest synthetic effect
    assert len(a) == 4


# ------------------------------------------------------------------ new evaluation helpers
@pytest.mark.parametrize("ties", [False, True])
def test_fast_auc_ap_matches_sklearn(ties):
    from sklearn.metrics import average_precision_score, roc_auc_score

    from symptompred.evaluate import _auc_ap
    rng = np.random.default_rng(1)
    y = rng.binomial(1, 0.2, 3000).astype(float)
    p = y * 0.5 + rng.random(3000)
    if ties:
        p = np.round(p, 1)
    u, inv = np.unique(p, return_inverse=True)
    auc, ap = _auc_ap(u, inv, y, np.ones(len(y)))
    assert auc == pytest.approx(roc_auc_score(y, p))
    assert ap == pytest.approx(average_precision_score(y, p))


def test_paired_bootstrap_detects_a_better_model():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 5000)
    signal = y + rng.normal(0, 1, 5000)
    better, worse = signal + rng.normal(0, 0.3, 5000), signal + rng.normal(0, 1.5, 5000)
    r = sp.paired_bootstrap(y, better, worse, n=300)
    assert r["delta_auroc"] > 0 and r["delta_auroc_lo"] > 0 and r["p_auroc"] < 0.01
    assert r["delta_auprc_lo"] > 0
    flipped = sp.paired_bootstrap(y, worse, better, n=300)
    assert flipped["delta_auroc"] == pytest.approx(-r["delta_auroc"]) and flipped["p_auroc"] > 0.99
    same = sp.paired_bootstrap(y, better, better, n=100)
    assert same["delta_auroc"] == 0 and same["delta_auroc_lo"] == 0 == same["delta_auroc_hi"]


def test_oracle_ceiling_beats_any_fitted_model(country_a):
    train, test = sp.temporal_split(country_a)
    ceiling = sp.oracle_ceiling(test, sp.CORE_FEATURES)
    model = sp.SymptomModel().fit(train).evaluate(test, bootstrap=0)
    assert ceiling["auroc"] >= model["auroc"] and ceiling["auprc"] >= model["auprc"]
    assert ceiling["combinations"] <= 2 ** 8


def test_model_factory_default_is_unchanged(two_countries):
    default = sp.compare(two_countries)
    explicit = sp.compare(two_countries, model_factory=sp.SymptomModel)
    pd.testing.assert_frame_equal(default, explicit)
    lr = sp.compare(two_countries, model_factory=sp.registry()["Logistic regression"])
    assert not np.allclose(lr["auroc"], default["auroc"])
    loo = sp.leave_one_out(two_countries, features="common", model_factory=sp.registry()["Naive Bayes"])
    assert len(loo) == 2


def test_pool_vs_specialist(two_countries):
    splits = {k: sp.temporal_split(v) for k, v in two_countries.items()}
    res = sp.pool_vs_specialist(splits, missing="absent")
    assert set(res["model"]) == {"specialist", "pooled, shared features", "pooled, all features",
                                 "pooled, all features + country"}
    assert len(res) == 8
    assert "male" not in res.attrs["shared_features"]
    y, p = res.attrs["predictions"][("b", "specialist")]
    np.testing.assert_array_equal(y, splits["b"][1]["y"].values)


def test_select_and_score_refuses_overlapping_periods(country_a):
    models = sp.registry()
    candidates = {"lr": models["Logistic regression"], "nb": models["Naive Bayes"]}
    with pytest.raises(AssertionError, match="leak"):
        sp.select_and_score(candidates, country_a, country_a)
    train, test = sp.temporal_split(country_a)
    best, table, model = sp.select_and_score(candidates, train, test, folds=3)
    assert best == table.index[0] and set(table.index) == {"lr", "nb"}
    assert model.n_train_ == len(train)

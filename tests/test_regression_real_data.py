"""Pin the headline numbers from notebook.ipynb / alternative-data.ipynb on the real datasets.

Slow (a few minutes): run with `pytest -m slow`. These catch any library change that silently
alters results.
"""

import warnings

import pytest

import symptompred as sp

from .conftest import requires_data

pytestmark = [pytest.mark.slow, requires_data]
TOL = 0.002


@pytest.fixture(scope="module")
def splits():
    israel = sp.datasets.israel.load(drop_unknown_sex=True)
    carbon = sp.datasets.carbon_health.load()
    out = {
        "israel": sp.date_split(israel, sp.datasets.israel.PAPER_TRAIN, sp.datasets.israel.PAPER_TEST),
        "carbon_health": sp.date_split(carbon, ("2020-04-01", "2020-08-31"), ("2020-09-01", "2020-10-31")),
    }
    if sp.datasets.brazil.available():
        brazil = sp.datasets.brazil.load()
        out["brazil"] = sp.date_split(brazil, ("2020-01-01", "2021-12-31"), ("2022-01-01", "2022-12-31"))
    return out


def test_paper_split_sizes(splits):
    train, test = splits["israel"]
    assert (len(train), len(test)) == (51831, 47401)
    assert (train.y.sum(), test.y.sum()) == (4769, 3624)


def test_paper_reproduction(splits):
    train, test = splits["israel"]
    r = sp.SymptomModel().fit(train).evaluate(test, bootstrap=0, sensitivity=0.873)
    assert r["auroc"] == pytest.approx(0.9016, abs=TOL)
    assert r["wp_sensitivity"] == pytest.approx(0.873, abs=0.001)
    assert r["wp_specificity"] == pytest.approx(0.720, abs=TOL)


def test_carbon_health_specialist(splits):
    train, test = splits["carbon_health"]
    r = sp.SymptomModel(features="all", missing="absent", n_models=5).fit(train).evaluate(test, bootstrap=0)
    assert r["auroc"] == pytest.approx(0.708, abs=TOL)


def test_pooling_numbers(splits):
    if "brazil" not in splits:
        pytest.skip("Brazil data not downloaded")
    pooled = sp.pool({name: tr for name, (tr, te) in splits.items()})
    common = sp.common_features(*(pd for pair in splits.values() for pd in pair))
    expected = {
        "all": ("all", {"israel": 0.901, "carbon_health": 0.704, "brazil": 0.639}),
        "shared": (common, {"israel": 0.777, "carbon_health": 0.678, "brazil": 0.691}),
    }
    for label, (features, scores) in expected.items():
        model = sp.SymptomModel(features=features, missing="absent", n_models=5).fit(pooled)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, (tr, te) in splits.items():
                auroc = model.evaluate(te.assign(source=name), bootstrap=0)["auroc"]
                assert auroc == pytest.approx(scores[name], abs=TOL), (label, name)


def test_pool_vs_specialist_reproduces_alternative_data_4_4(splits):
    if "brazil" not in splits:
        pytest.skip("Brazil data not downloaded")
    res = sp.pool_vs_specialist(splits, variants=("shared", "all"), missing="absent", n_models=5)
    table = res.pivot(index="model", columns="test", values="auroc")
    expected = {"pooled, all features": {"israel": 0.901, "carbon_health": 0.704, "brazil": 0.639},
                "pooled, shared features": {"israel": 0.777, "carbon_health": 0.678, "brazil": 0.691},
                "specialist": {"israel": 0.901, "carbon_health": 0.708, "brazil": 0.636}}
    for model, scores in expected.items():
        for country, value in scores.items():
            assert table.loc[model, country] == pytest.approx(value, abs=TOL), (model, country)

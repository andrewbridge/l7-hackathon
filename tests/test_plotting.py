import matplotlib.pyplot as plt
import numpy as np
import pytest

import symptompred as sp


@pytest.fixture
def curves():
    rng = np.random.default_rng(0)
    y = rng.binomial(1, 0.2, 500)
    return {"good": (y, y + rng.random(500)), "noise": (y, rng.random(500))}


def test_plot_curves(curves):
    ax_roc, ax_pr = sp.plot_curves(curves, title="test")
    labels = [t.get_text() for t in ax_roc.get_legend().get_texts()]
    assert labels[0].startswith("good (") and len(labels) == 2
    assert ax_roc.get_title() == "test: ROC"
    plt.close("all")


def test_plot_curve_grid(curves):
    fig = sp.plot_curve_grid({"x": curves, "y": curves}, suptitle="grid")
    assert len(fig.axes) == 4
    plt.close(fig)


def test_plot_shap_grid_marks_features_not_collected(country_a, country_b):
    expl = {"a": sp.SymptomModel(num_boost_round=20).fit(country_a).shap_values(country_a.iloc[:300]),
            "b": sp.SymptomModel(features="all", num_boost_round=20).fit(country_b).shap_values(country_b.iloc[:300])}
    order = sp.shap_feature_order(expl)
    assert set(order) == set(expl["a"].feature_names) | set(expl["b"].feature_names)
    fig = sp.plot_shap_grid(expl, max_points=200)
    texts = [t.get_text() for ax in fig.axes for t in ax.texts]
    assert texts.count("not collected") == 2  # b has no `male`, a has no `loss_of_smell`
    plt.close(fig)

"""symptompred: predict COVID-19 test results from symptoms, across datasets.

Reimplements Zoabi et al. (2021, npj Digital Medicine) as a reusable library. Every dataset is
converted into one convention (see `schema`), so the same model and evaluation run on any of them:

    import symptompred as sp

    israel = sp.datasets.israel.load()
    carbon = sp.datasets.carbon_health.load()

    model = sp.SymptomModel().fit(israel)
    model.evaluate(carbon)

    sp.compare({"israel": israel, "carbon_health": carbon})      # train on each, test on each
    sp.leave_one_out({...})                                       # pool others, test on held-out one

    models = sp.registry()                                        # alternative model types
    sp.compare({...}, model_factory=models["CatBoost"])

To add a dataset, write a loader that returns a DataFrame with `date`, `y` and feature columns
named from `sp.ALL_FEATURES`, then pass it to `validate()`.
"""

from . import datasets
from .alternatives import (FAMILIES, TUNED_PARAMS, CellRateClassifier, EnsembleModel, ScorecardClassifier,
                           registry)
from .evaluate import (bootstrap_ci, compare, cross_validate, date_split, forward_selection, leave_one_out, metrics,
                       oracle_ceiling, paired_bootstrap, pivot, pool_vs_specialist, select_and_score, temporal_split,
                       working_point)
from .model import PAPER_PARAMS, BaseSymptomModel, EstimatorModel, SymptomModel, recency_weights
from .plotting import plot_curve_grid, plot_curves, plot_shap_grid, shap_feature_order
from .schema import (ALL_FEATURES, CORE_FEATURES, DEMOGRAPHIC_FEATURES, EXTRA_FEATURES, SchemaError, available_features,
                     common_features, feature_summary, pool, validate)

__all__ = [
    "SymptomModel", "PAPER_PARAMS", "datasets",
    "BaseSymptomModel", "EstimatorModel", "EnsembleModel", "CellRateClassifier", "ScorecardClassifier", "registry",
    "FAMILIES", "TUNED_PARAMS", "recency_weights",
    "paired_bootstrap", "oracle_ceiling", "cross_validate", "select_and_score", "pool_vs_specialist",
    "compare", "leave_one_out", "forward_selection", "plot_curves", "plot_curve_grid", "plot_shap_grid", "shap_feature_order", "pivot", "pool", "metrics", "bootstrap_ci", "working_point", "temporal_split", "date_split",
    "validate", "available_features", "common_features", "feature_summary", "SchemaError",
    "CORE_FEATURES", "EXTRA_FEATURES", "ALL_FEATURES", "DEMOGRAPHIC_FEATURES",
]

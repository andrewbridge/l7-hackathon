"""Alternative model types to the paper's LightGBM, in one place so every experiment uses the same definitions.

    import symptompred as sp

    models = sp.registry()                        # name -> factory
    m = models["Logistic regression"](features="all", missing="absent").fit(train)
    sp.compare(datasets, model_factory=models["CatBoost"])

Every factory accepts the usual model options (`features`, `missing`, `n_models`, `seed`,
`source_feature`) and `params`, a dict of hyperparameters for the underlying estimator.

Two small estimators that don't exist off the shelf are defined here:

* `CellRateClassifier`: a lookup table. With 8 yes/no answers there are only 256 possible inputs, so
  the positive rate among training rows with exactly the same answers is a natural predictor. Rare
  combinations are shrunk towards a simpler model's prediction (empirical Bayes).
* `ScorecardClassifier`: logistic regression rounded to whole points, like a clinical risk score
  ("contact +6, fever +3, ...; add them up").
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from .model import BaseSymptomModel, EstimatorModel, SymptomModel
from .schema import validate

# The groupmate's Optuna-tuned LightGBM (github.com/warren-davies4/covid-prediction-hackathon,
# tune_hyperparameters_results.json). They chose the number of rounds by early stopping (patience 30).
TUNED_PARAMS = {
    "num_leaves": 7,
    "min_data_in_leaf": 59,
    "feature_fraction": 0.7154446959919848,
    "bagging_fraction": 0.9716985486424213,
    "bagging_freq": 4,
    "learning_rate": 0.14288365717831378,
    "lambda_l1": 0.23836096183964298,
    "lambda_l2": 0.08902319141094964,
}


def _with_weights(estimator, X, y, sample_weight):
    if sample_weight is None:
        return estimator.fit(X, y)
    return estimator.fit(X, y, sample_weight=sample_weight)


class CellRateClassifier(ClassifierMixin, BaseEstimator):
    """Positive rate per exact input combination, shrunk towards a backoff model.

    p(cell) = (positives + alpha * backoff(cell)) / (rows + alpha)

    alpha : pseudo-count. 0 = raw rates (unseen combinations get the backoff); large = the backoff model.
    backoff : classifier for the prior (default: logistic regression, i.e. main effects only).
    """

    def __init__(self, alpha: float = 20.0, backoff=None):
        self.alpha = alpha
        self.backoff = backoff

    @staticmethod
    def _keys(X: np.ndarray) -> list[bytes]:
        return [row.tobytes() for row in np.nan_to_num(X, nan=-1.0)]

    def fit(self, X, y, sample_weight=None):
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, dtype=float)
        self.backoff_ = _with_weights(clone(self.backoff or LogisticRegression(max_iter=2000)), X, y, sample_weight)
        cells, first, inverse = np.unique(np.nan_to_num(X, nan=-1.0), axis=0, return_index=True, return_inverse=True)
        inverse = inverse.ravel()
        self.cell_n_ = np.bincount(inverse, weights=w)
        self.cell_pos_ = np.bincount(inverse, weights=w * y)
        self.table_ = dict(zip(self._keys(X[first]), zip(self.cell_pos_, self.cell_n_)))
        self.classes_ = np.array([0, 1])
        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=float)
        prior = self.backoff_.predict_proba(X)[:, 1]
        p = prior.copy()
        for i, key in enumerate(self._keys(X)):
            if key in self.table_:
                pos, n = self.table_[key]
                p[i] = (pos + self.alpha * prior[i]) / (n + self.alpha) if n + self.alpha > 0 else prior[i]
        return np.column_stack([1 - p, p])


class ScorecardClassifier(ClassifierMixin, BaseEstimator):
    """Logistic regression with coefficients rounded to integer points (largest effect = `max_points`).

    The points total is mapped back to a probability with a one-variable logistic regression.
    `points_` holds the points per input column.
    """

    def __init__(self, max_points: int = 10, C: float = 1.0):
        self.max_points = max_points
        self.C = C

    def fit(self, X, y, sample_weight=None):
        names = list(X.columns) if hasattr(X, "columns") else [f"x{i}" for i in range(np.shape(X)[1])]
        X, y = np.asarray(X, dtype=float), np.asarray(y)
        lr = _with_weights(LogisticRegression(C=self.C, max_iter=2000), X, y, sample_weight)
        scale = self.max_points / np.abs(lr.coef_[0]).max()
        self.points_ = pd.Series(np.round(lr.coef_[0] * scale).astype(int), index=names)
        self.calibration_ = _with_weights(LogisticRegression(), self._score(X), y, sample_weight)
        self.classes_ = np.array([0, 1])
        return self

    def _score(self, X):
        return (np.asarray(X, dtype=float) @ self.points_.values)[:, None]

    def predict_proba(self, X):
        return self.calibration_.predict_proba(self._score(X))


class EnsembleModel(BaseSymptomModel):
    """Combine several models: average their log-odds, or stack them with a logistic regression.

    members : {name: factory}, e.g. a subset of `registry()`. Each factory gets `model_kwargs`.
    method : "average", or "stack" (a logistic regression on the members' out-of-fold log-odds,
        from `folds`-fold cross-validation on the training data).
    """

    def __init__(self, members: dict[str, Callable], method: str = "average", folds: int = 5, seed: int = 0,
                 **model_kwargs):
        if method not in ("average", "stack"):
            raise ValueError("method must be 'average' or 'stack'")
        self.members = members
        self.method = method
        self.folds = folds
        self.seed = seed
        self.model_kwargs = model_kwargs

    def __repr__(self) -> str:
        return f"EnsembleModel({list(self.members)}, method={self.method!r})"

    def _make(self, name):
        return self.members[name](seed=self.seed, **self.model_kwargs)

    def fit(self, df: pd.DataFrame, sample_weight=None):
        df = validate(df)
        self.models_ = {name: self._make(name).fit(df, sample_weight=sample_weight) for name in self.members}
        self.features_ = next(iter(self.models_.values())).features_
        self.n_train_ = len(df)
        if self.method == "stack":
            oof = np.zeros((len(df), len(self.members)))
            weights = None if sample_weight is None or isinstance(sample_weight, str) else np.asarray(sample_weight)
            for tr, va in StratifiedKFold(self.folds, shuffle=True, random_state=self.seed).split(df, df["y"]):
                for j, name in enumerate(self.members):
                    m = self._make(name).fit(df.iloc[tr], sample_weight=sample_weight if weights is None
                                             else weights[tr])
                    oof[va, j] = m.predict_raw(df.iloc[va])
            self.meta_ = LogisticRegression(max_iter=2000).fit(oof, df["y"].values)
        self.is_fitted_ = True
        return self

    def _member_log_odds(self, df):
        return np.column_stack([m.predict_raw(df) for m in self.models_.values()])

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        z = self._member_log_odds(df)
        return self.meta_.decision_function(z) if self.method == "stack" else z.mean(axis=1)


# ----------------------------------------------------------------------------- registry
FAMILIES = {
    "LightGBM (paper)": "boosting (benchmark)",
    "LightGBM (tuned params)": "boosting (benchmark)",
    "Logistic regression": "linear",
    "Logistic + interactions": "linear",
    "Scorecard": "linear",
    "Naive Bayes": "probabilistic",
    "Cell-rate lookup": "lookup",
    "k-nearest neighbours": "lookup",
    "Decision tree": "tree",
    "Random forest": "bagging",
    "Extra trees": "bagging",
    "Neural network (MLP)": "neural",
    "XGBoost": "boosting",
    "CatBoost": "boosting",
    "HistGradientBoosting": "boosting",
}


def _estimator_factory(make: Callable, missing_indicators: bool = True) -> Callable:
    def factory(params: dict | None = None, **model_kwargs):
        return EstimatorModel(make().set_params(**(params or {})), missing_indicators=missing_indicators,
                              **model_kwargs)
    return factory


def registry() -> dict[str, Callable]:
    """Every candidate model, as name -> factory(params=None, **model_kwargs). See `FAMILIES` for groupings."""
    from catboost import CatBoostClassifier
    from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.tree import DecisionTreeClassifier
    from xgboost import XGBClassifier

    forest = dict(n_estimators=300, min_samples_leaf=20, max_features="sqrt", n_jobs=-1)
    return {
        "LightGBM (paper)": lambda params=None, **kw: SymptomModel(params=params, **kw),
        "LightGBM (tuned params)": lambda params=None, **kw: SymptomModel(
            params={**TUNED_PARAMS, **(params or {})}, early_stopping_rounds=30, **kw),
        "Logistic regression": _estimator_factory(lambda: LogisticRegression(max_iter=2000)),
        "Logistic + interactions": _estimator_factory(lambda: Pipeline([
            ("pairs", PolynomialFeatures(2, interaction_only=True, include_bias=False)),
            ("lr", LogisticRegression(C=1.0, max_iter=5000))])),
        "Scorecard": _estimator_factory(ScorecardClassifier),
        "Naive Bayes": _estimator_factory(BernoulliNB),
        "Cell-rate lookup": _estimator_factory(CellRateClassifier),
        "k-nearest neighbours": _estimator_factory(
            lambda: KNeighborsClassifier(n_neighbors=200, metric="hamming", algorithm="brute")),
        "Decision tree": _estimator_factory(lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=50)),
        "Random forest": _estimator_factory(lambda: RandomForestClassifier(**forest)),
        "Extra trees": _estimator_factory(lambda: ExtraTreesClassifier(**forest)),
        "Neural network (MLP)": _estimator_factory(lambda: MLPClassifier(
            hidden_layer_sizes=(32, 16), alpha=1e-3, early_stopping=True, max_iter=300)),
        "XGBoost": _estimator_factory(lambda: XGBClassifier(
            n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8,
            n_jobs=-1, verbosity=0), missing_indicators=False),
        "CatBoost": _estimator_factory(lambda: CatBoostClassifier(
            iterations=500, depth=4, learning_rate=0.05, verbose=0, thread_count=-1, allow_writing_files=False),
            missing_indicators=False),
        "HistGradientBoosting": _estimator_factory(lambda: HistGradientBoostingClassifier(
            learning_rate=0.05, max_iter=300, max_leaf_nodes=15), missing_indicators=False),
    }

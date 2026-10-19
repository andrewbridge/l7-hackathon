"""Models that predict a COVID-19 test result from schema-format data.

* `SymptomModel`: the paper's LightGBM approach.
* `EstimatorModel`: any scikit-learn-style classifier (logistic regression, XGBoost, ...), with the same
  feature, missing-value and pooling handling, so alternatives can be compared like-for-like.

Both share `BaseSymptomModel`, which turns a schema DataFrame into a feature matrix.
"""

from __future__ import annotations

import warnings
from typing import Iterable

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .evaluate import bootstrap_ci, metrics, working_point
from .schema import ALL_FEATURES, CORE_FEATURES, DEMOGRAPHIC_FEATURES, available_features, validate

# Hyperparameters from covidpred-master/hyperparameters.txt
PAPER_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "num_leaves": 20,
    "min_data_in_leaf": 4,
    "feature_fraction": 0.2,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "learning_rate": 0.05,
    "verbose": -1,
}
SOURCE_PREFIX = "source_"


class BaseSymptomModel:
    """Shared plumbing: feature selection, missing values, sample weights, country indicators, evaluation.

    Parameters
    ----------
    features : "core", "all", or a list of names
        "core" = the paper's eight features (whichever the training data has); "all" = every
        available feature including extras like loss of smell.
    missing : "native" or "absent"
        "native" (paper): the model sees missing values as missing, so "not recorded" can become a signal.
        "absent": treat an unrecorded symptom or contact as "not reported" (0). Age and sex stay
        missing. Use this when missingness reflects the testing setting rather than the patient, as in
        Carbon Health, where "fever not recorded" meant *higher* risk in spring and *lower* in autumn.
    n_models :
        Train this many models on different seeds and average them. 1 = the paper's approach.
    source_feature :
        Add one 0/1 column per training `source` (country), so a pooled model can learn country-specific
        adjustments directly. A country not seen in training gets all zeros.
    """

    def __init__(self, features: str | Iterable[str] = "core", missing: str = "native", n_models: int = 1,
                 seed: int = 0, source_feature: bool = False):
        if missing not in ("native", "absent"):
            raise ValueError("missing must be 'native' or 'absent'")
        self.features = features
        self.missing = missing
        self.n_models = n_models
        self.seed = seed
        self.source_feature = source_feature

    # ------------------------------------------------------------------ fitting
    def _resolve_features(self, df: pd.DataFrame) -> list[str]:
        if self.features == "core":
            feats = available_features(df, CORE_FEATURES)
        elif self.features == "all":
            feats = available_features(df, ALL_FEATURES)
        else:
            feats = list(self.features)
            absent = [f for f in feats if f not in available_features(df)]
            if absent:
                raise ValueError(f"Requested feature(s) not available in training data: {absent}")
        if not feats:
            raise ValueError("No usable features in the training data.")
        return feats

    def _handle_missing(self, df: pd.DataFrame) -> pd.DataFrame:
        """With missing="absent", fill blanks with 0, but only for features the row's source collected.

        Pooled data mixes "never collected" (e.g. Israel has no loss_of_smell column at all) with "collected
        but blank for this row". Only the second means "not reported", so filling is done per `source`.
        """
        if self.missing != "absent":
            return df
        fillable = [f for f in self.features_ if f in df.columns and f not in DEMOGRAPHIC_FEATURES]
        groups = df.groupby("source").groups.values() if "source" in df.columns else [df.index]
        df = df.copy()
        for idx in groups:
            collected = [f for f in fillable if df.loc[idx, f].notna().any()]
            df.loc[idx, collected] = df.loc[idx, collected].fillna(0.0)
        return df

    @staticmethod
    def _weights(df: pd.DataFrame, sample_weight) -> pd.Series | None:
        if sample_weight is None:
            return None
        if isinstance(sample_weight, str):
            if sample_weight != "balance_sources" or "source" not in df.columns:
                raise ValueError('sample_weight="balance_sources" needs a `source` column')
            counts = df["source"].map(df["source"].value_counts())
            sample_weight = (len(df) / df["source"].nunique() / counts).values
        return pd.Series(np.asarray(sample_weight, dtype=float), index=df.index)

    def _design(self, df: pd.DataFrame) -> pd.DataFrame:
        """The model's input columns: the features, plus country indicators if `source_feature`."""
        X = df[self.features_]
        if self.sources_:
            source = df["source"] if "source" in df.columns else pd.Series(None, index=df.index)
            X = X.assign(**{f"{SOURCE_PREFIX}{s}": (source == s).astype(float) for s in self.sources_})
        return X

    def fit(self, df: pd.DataFrame, sample_weight: str | np.ndarray | None = None):
        """Fit on schema-format data.

        sample_weight : None, "balance_sources", or an array (one weight per row).
            "balance_sources" gives each `source` the same total weight, so pooling a large dataset with a
            small one doesn't let the large one dominate.
        """
        df = validate(df)
        self.features_ = self._resolve_features(df)
        df = self._handle_missing(df)
        self.n_train_ = len(df)
        if self.source_feature and "source" not in df.columns:
            raise ValueError("source_feature=True needs a `source` column (e.g. from symptompred.pool)")
        self.sources_ = sorted(df["source"].unique()) if self.source_feature else []
        # Stratify any internal splits by outcome (and source, when pooling) so every part is represented
        strata = df["y"].astype(str) + ("_" + df["source"].astype(str) if "source" in df.columns else "")
        self._fit(self._design(df), df["y"], self._weights(df, sample_weight), strata)
        self.is_fitted_ = True
        return self

    def _fit(self, X: pd.DataFrame, y: pd.Series, weights: pd.Series | None, strata: pd.Series):
        raise NotImplementedError

    # --------------------------------------------------------------- predicting
    def _check_fitted(self):
        if not getattr(self, "is_fitted_", False):
            raise RuntimeError("Call fit() first.")

    def _X(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._handle_missing(validate(df, require_y=False))
        # Features this data never collected stay NaN ("unknown"), never filled as "not reported"
        absent = [f for f in self.features_ if f not in available_features(df)]
        if absent:
            warnings.warn(
                f"Feature(s) {absent} not in this data; using NaN. Features that were never missing in "
                "training may be treated by the model as 0 when missing, so predictions may be biased. Prefer "
                "training on symptompred.common_features(train, test), or pooled data where they're sometimes "
                "absent.", stacklevel=3)
            df = df.assign(**{f: np.nan for f in absent if f not in df.columns})
        return self._design(df)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Log-odds of a positive test."""
        raise NotImplementedError

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of a positive test."""
        return 1 / (1 + np.exp(-self.predict_raw(df)))

    # --------------------------------------------------------------- evaluating
    def evaluate(self, df: pd.DataFrame, bootstrap: int = 1000, sensitivity: float | None = None) -> pd.Series:
        """auROC, auPRC, Brier score etc. on labelled data, with optional bootstrap CIs and a working point."""
        df = validate(df)
        p = self.predict(df)
        out = metrics(df["y"].values, p)
        if bootstrap:
            out.update(bootstrap_ci(df["y"].values, p, n=bootstrap, seed=self.seed))
        if sensitivity is not None:
            out.update({f"wp_{k}": v for k, v in working_point(df["y"].values, p, sensitivity).items()})
        return pd.Series(out)


class SymptomModel(BaseSymptomModel):
    """Predict a positive COVID-19 test from binary symptom / demographic features (the paper's LightGBM).

    Parameters
    ----------
    features, missing, n_models, seed, source_feature : see `BaseSymptomModel`.
    params : dict, optional
        LightGBM parameters, merged over PAPER_PARAMS.
    num_boost_round, early_stopping_rounds :
        Paper: 603 and 5. We default to 50 for early stopping: with 5, results swing between
        ~0.76 and ~0.90 auROC depending on the seed (see notebook.ipynb section 15).
    validation_fraction :
        Share of the training data held out for early stopping (paper: 0.2, i.e. a 4:1 split).
    """

    def __init__(self, features: str | Iterable[str] = "core", params: dict | None = None,
                 num_boost_round: int = 603, early_stopping_rounds: int = 50,
                 validation_fraction: float = 0.2, n_models: int = 1, seed: int = 0,
                 missing: str = "native", source_feature: bool = False):
        super().__init__(features=features, missing=missing, n_models=n_models, seed=seed,
                         source_feature=source_feature)
        self.params = {**PAPER_PARAMS, **(params or {})}
        self.num_boost_round = num_boost_round
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_fraction = validation_fraction

    @classmethod
    def paper(cls, **kwargs) -> "SymptomModel":
        """Exactly the paper's settings (single model, early stopping after 5 rounds)."""
        return cls(**{"early_stopping_rounds": 5, "n_models": 1, **kwargs})

    def __repr__(self) -> str:
        fitted = f", fitted on {self.n_train_:,} rows" if hasattr(self, "boosters_") else ""
        return f"SymptomModel(features={self.features!r}, n_models={self.n_models}, missing={self.missing!r}{fitted})"

    def _fit(self, X, y, weights, strata):
        weights = weights if weights is not None else pd.Series(1.0, index=X.index)
        self.boosters_ = []
        for i in range(self.n_models):
            seed = self.seed + i
            tr, va = train_test_split(X, test_size=self.validation_fraction, random_state=seed, stratify=strata)
            booster = lgb.train(
                {**self.params, "seed": seed},
                lgb.Dataset(tr, y[tr.index], weight=weights[tr.index]),
                num_boost_round=self.num_boost_round,
                valid_sets=[lgb.Dataset(va, y[va.index], weight=weights[va.index])],
                callbacks=[lgb.early_stopping(self.early_stopping_rounds, verbose=False)],
            )
            self.boosters_.append(booster)

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Log-odds averaged across the ensemble."""
        self._check_fitted()
        X = self._X(df)
        return np.mean([b.predict(X, num_iteration=b.best_iteration, raw_score=True) for b in self.boosters_], axis=0)

    def shap_values(self, df: pd.DataFrame):
        """SHAP values (log-odds scale) as a shap.Explanation, averaged across the ensemble."""
        import shap  # imported lazily: optional and slow to import

        self._check_fitted()
        X = self._X(df)
        contrib = np.mean([b.predict(X, num_iteration=b.best_iteration, pred_contrib=True) for b in self.boosters_], axis=0)
        return shap.Explanation(values=contrib[:, :-1], base_values=contrib[:, -1],
                                data=X.values, feature_names=list(X.columns))

    @property
    def n_trees_(self) -> list[int]:
        self._check_fitted()
        return [b.best_iteration for b in self.boosters_]


# ------------------------------------------------------------------- any sklearn-style classifier
def _set_seed(estimator, seed: int):
    """Set every random_state the estimator (or pipeline step) exposes; CatBoost calls it random_seed."""
    params = estimator.get_params(deep=True)
    keys = {k: seed for k in params if k == "random_state" or k.endswith("__random_state")}
    if type(estimator).__name__.startswith("CatBoost"):
        keys = {"random_seed": seed}
    if keys:
        estimator.set_params(**keys)
    return estimator


def _fit_kwargs(estimator, weights: pd.Series | None) -> dict:
    import inspect

    if weights is None:
        return {}
    final = estimator.steps[-1][1] if hasattr(estimator, "steps") else estimator
    if "sample_weight" not in inspect.signature(final.fit).parameters:
        warnings.warn(f"{type(final).__name__} doesn't support sample weights; they are ignored.", stacklevel=4)
        return {}
    if hasattr(estimator, "steps"):  # sklearn Pipeline: route the weights to the final step
        return {f"{estimator.steps[-1][0]}__sample_weight": weights.values}
    return {"sample_weight": weights.values}


def _unique_rows(X: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    """Distinct rows of X (NaN-aware) and, for every row of X, which distinct row it is.

    Symptom data has only a few hundred distinct inputs, so predicting on those and mapping back is
    exact and much faster (it makes k-nearest neighbours and SHAP practical).
    """
    codes = X.groupby(list(X.columns), dropna=False, sort=False).ngroup().values
    first = ~pd.Series(codes).duplicated().values
    return X[first], codes


class EstimatorModel(BaseSymptomModel):
    """Wrap any scikit-learn-style classifier (`fit`, `predict_proba`) in the symptompred workflow.

    Parameters
    ----------
    estimator : an unfitted classifier, e.g. `LogisticRegression()` or `xgboost.XGBClassifier()`.
        It's cloned, so the same object can be reused.
    features, missing, n_models, seed, source_feature : see `BaseSymptomModel`.
    missing_indicators :
        True: missing values become 0 plus a `<feature>_missing` 0/1 column (for each feature with any
        missing values in training). Needed for models that can't take NaN (linear, Naive Bayes, kNN,
        neural networks). False: pass NaN through, for models that handle it natively (XGBoost, CatBoost,
        HistGradientBoosting).
    """

    def __init__(self, estimator, features: str | Iterable[str] = "core", missing: str = "native",
                 missing_indicators: bool = True, n_models: int = 1, seed: int = 0, source_feature: bool = False):
        super().__init__(features=features, missing=missing, n_models=n_models, seed=seed,
                         source_feature=source_feature)
        self.estimator = estimator
        self.missing_indicators = missing_indicators

    def __repr__(self) -> str:
        fitted = f", fitted on {self.n_train_:,} rows" if getattr(self, "is_fitted_", False) else ""
        return (f"EstimatorModel({type(self.estimator).__name__}, features={self.features!r}, "
                f"missing={self.missing!r}{fitted})")

    def _prepare(self, X: pd.DataFrame) -> pd.DataFrame:
        if not self.missing_indicators:
            return X
        indicators = {f"{f}_missing": X[f].isna().astype(float) for f in self.indicator_features_}
        return X.fillna(0.0).assign(**indicators)

    def _fit(self, X, y, weights, strata):
        from sklearn.base import clone

        self.indicator_features_ = [f for f in self.features_ if X[f].isna().any()] if self.missing_indicators else []
        X = self._prepare(X)
        self.columns_ = list(X.columns)
        self.background_ = X.sample(min(200, len(X)), random_state=self.seed)  # for SHAP
        self.estimators_ = []
        for i in range(self.n_models):
            est = _set_seed(clone(self.estimator), self.seed + i)
            est.fit(X, y.values, **_fit_kwargs(est, weights))
            self.estimators_.append(est)

    def _proba(self, X: pd.DataFrame) -> np.ndarray:
        """Mean predicted probability across the ensemble, for an already-prepared matrix."""
        unique, codes = _unique_rows(X)
        p = np.mean([est.predict_proba(unique)[:, 1] for est in self.estimators_], axis=0)
        return p[codes]

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        """Probability of a positive test (averaged across the ensemble)."""
        self._check_fitted()
        return self._proba(self._prepare(self._X(df)))

    def predict_raw(self, df: pd.DataFrame) -> np.ndarray:
        """Log-odds of `predict` (clipped, since some models predict exactly 0 or 1)."""
        p = np.clip(self.predict(df), 1e-9, 1 - 1e-9)
        return np.log(p / (1 - p))

    def shap_values(self, df: pd.DataFrame, background: int = 100):
        """SHAP values (log-odds scale) as a shap.Explanation, for any estimator (model-agnostic).

        Uses the exact algorithm when there are at most 10 input columns (permutation otherwise), on the
        distinct rows only. A `<feature>_missing` indicator's contribution is added to its feature, so the
        result has one column per feature (plus country indicators), like `SymptomModel.shap_values`.
        """
        import shap

        self._check_fitted()
        raw = self._X(df)
        X = self._prepare(raw)
        unique, codes = _unique_rows(X)

        def log_odds(arr):
            p = np.clip(self._proba(pd.DataFrame(arr, columns=self.columns_)), 1e-9, 1 - 1e-9)
            return np.log(p / (1 - p))

        masker = shap.maskers.Independent(self.background_.values, max_samples=background)
        algorithm = "exact" if len(self.columns_) <= 10 else "permutation"
        expl = shap.Explainer(log_odds, masker, algorithm=algorithm, seed=self.seed)(
            unique.values, **({} if algorithm == "exact" else {"max_evals": 2 * len(self.columns_) + 1}))
        values = pd.DataFrame(expl.values, columns=self.columns_)
        for f in self.indicator_features_:
            values[f] += values.pop(f"{f}_missing")
        names = list(raw.columns)
        return shap.Explanation(values=values[names].values[codes], base_values=np.asarray(expl.base_values)[codes],
                                data=raw.values, feature_names=names)


def recency_weights(df: pd.DataFrame, half_life_days: float) -> np.ndarray:
    """Sample weights that halve every `half_life_days` back from the latest date (per `source` if present)."""
    latest = df.groupby("source")["date"].transform("max") if "source" in df.columns else df["date"].max()
    age = (latest - df["date"]).dt.days.values
    return 0.5 ** (age / half_life_days)

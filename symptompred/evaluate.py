"""Metrics, splits, and cross-dataset comparison."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .schema import ALL_FEATURES, common_features, validate


class Predictions(dict):
    """{key: (y_true, y_score)} stored in a results DataFrame's `.attrs["predictions"]`.

    Compares by identity: pandas compares `.attrs` when concatenating DataFrames, and comparing the
    NumPy arrays inside would raise. (Different results' predictions are simply dropped from a concat.)
    """
    __eq__ = object.__eq__
    __hash__ = object.__hash__


# --------------------------------------------------------------------------- metrics
def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    y, p = np.asarray(y), np.asarray(p)
    two_classes = len(np.unique(y)) == 2
    return {
        "n": len(y),
        "positives": int(y.sum()),
        "prevalence": y.mean(),
        "auroc": roc_auc_score(y, p) if two_classes else np.nan,
        "auprc": average_precision_score(y, p) if two_classes else np.nan,
        "brier": np.mean((p - y) ** 2),
        "mean_predicted": p.mean(),
    }


def bootstrap_ci(y: np.ndarray, p: np.ndarray, n: int = 1000, seed: int = 0, level: float = 0.95) -> dict:
    """Percentile bootstrap confidence intervals for auROC and auPRC (the paper's method)."""
    y, p = np.asarray(y), np.asarray(p)
    rng = np.random.default_rng(seed)
    aucs, prcs = [], []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].min() == y[idx].max():
            continue
        aucs.append(roc_auc_score(y[idx], p[idx]))
        prcs.append(average_precision_score(y[idx], p[idx]))
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"auroc_lo": np.percentile(aucs, lo), "auroc_hi": np.percentile(aucs, hi),
            "auprc_lo": np.percentile(prcs, lo), "auprc_hi": np.percentile(prcs, hi)}


def _auc_ap(order: np.ndarray, inverse: np.ndarray, y: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    """Weighted auROC and average precision, grouping tied scores (matches scikit-learn on unit weights).

    `order`/`inverse` come from `np.unique(scores, return_inverse=True)`, so the work per call is a couple
    of `bincount`s: fast enough for thousands of bootstrap resamples.
    """
    pos = np.bincount(inverse, weights=weights * y, minlength=len(order))
    neg = np.bincount(inverse, weights=weights * (1 - y), minlength=len(order))
    P, N = pos.sum(), neg.sum()
    auc = (pos * (np.cumsum(neg) - neg + 0.5 * neg)).sum() / (P * N)
    tp, fp = np.cumsum(pos[::-1]), np.cumsum(neg[::-1])
    gained = np.diff(np.concatenate([[0.0], tp])) > 0  # thresholds where recall increases
    ap = (np.diff(np.concatenate([[0.0], tp]))[gained] / P * tp[gained] / (tp[gained] + fp[gained])).sum()
    return auc, ap


def paired_bootstrap(y: np.ndarray, p_a: np.ndarray, p_b: np.ndarray, n: int = 2000, seed: int = 0,
                     level: float = 0.95) -> dict:
    """Is model A better than model B on the same test rows? Paired bootstrap of the auROC / auPRC difference.

    Each resample draws the same rows for both models, so the shared noise of the test set cancels out
    and much smaller differences can be detected than by comparing two separate confidence intervals.

    Returns A's and B's scores, the difference (A - B) with its CI, and `p_*`: the share of resamples in
    which A was *not* better (a one-sided p-value; small = A is reliably better).
    """
    y = np.asarray(y, dtype=float)
    _, inv_a = np.unique(np.asarray(p_a), return_inverse=True)
    _, inv_b = np.unique(np.asarray(p_b), return_inverse=True)
    ua, ub = np.arange(inv_a.max() + 1), np.arange(inv_b.max() + 1)
    full = np.ones(len(y))
    (auc_a, ap_a), (auc_b, ap_b) = _auc_ap(ua, inv_a, y, full), _auc_ap(ub, inv_b, y, full)
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        w = np.bincount(rng.integers(0, len(y), len(y)), minlength=len(y)).astype(float)
        if (w * y).sum() == 0 or (w * (1 - y)).sum() == 0:
            continue
        a, b = _auc_ap(ua, inv_a, y, w), _auc_ap(ub, inv_b, y, w)
        deltas.append((a[0] - b[0], a[1] - b[1]))
    d = np.array(deltas)
    lo, hi = (1 - level) / 2 * 100, (1 + level) / 2 * 100
    return {"auroc_a": auc_a, "auroc_b": auc_b, "delta_auroc": auc_a - auc_b,
            "delta_auroc_lo": np.percentile(d[:, 0], lo), "delta_auroc_hi": np.percentile(d[:, 0], hi),
            "p_auroc": (d[:, 0] <= 0).mean(),
            "auprc_a": ap_a, "auprc_b": ap_b, "delta_auprc": ap_a - ap_b,
            "delta_auprc_lo": np.percentile(d[:, 1], lo), "delta_auprc_hi": np.percentile(d[:, 1], hi),
            "p_auprc": (d[:, 1] <= 0).mean()}


def working_point(y: np.ndarray, p: np.ndarray, sensitivity: float) -> dict:
    """The highest threshold that reaches at least `sensitivity`, and what it costs."""
    y, p = np.asarray(y), np.asarray(p)
    for t in np.sort(np.unique(p))[::-1]:
        pred = p >= t
        sens = pred[y == 1].mean()
        if sens >= sensitivity:
            tp, fp = (pred & (y == 1)).sum(), (pred & (y == 0)).sum()
            tn, fn = (~pred & (y == 0)).sum(), (~pred & (y == 1)).sum()
            return {"threshold": t, "sensitivity": sens, "specificity": tn / (tn + fp),
                    "ppv": tp / (tp + fp), "npv": tn / (tn + fn) if tn + fn else np.nan, "flagged": pred.mean()}
    return {}


# ---------------------------------------------------------------------------- splits
def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Train on earlier dates, test on the latest ~`test_fraction` of rows (split on a date boundary)."""
    df = df.sort_values("date")
    cutoff = df["date"].iloc[int(len(df) * (1 - test_fraction))]
    return df[df["date"] < cutoff], df[df["date"] >= cutoff]


def date_split(df: pd.DataFrame, train: tuple, test: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Explicit inclusive date windows, e.g. the paper's train=("2020-03-22", "2020-03-31")."""
    return df[df["date"].between(*train)], df[df["date"].between(*test)]


# ------------------------------------------------------------------------ comparison
def compare(datasets: dict[str, pd.DataFrame], features="common", splits: dict | None = None,
            holdout: dict[str, pd.DataFrame] | None = None, bootstrap: int = 0, unseen_on: str = "full",
            model_factory=None, **model_kwargs) -> pd.DataFrame:
    """Train on each dataset, test on its own held-out period and on every other dataset.

    Parameters
    ----------
    datasets : name -> DataFrame (schema format). Each is used for training and testing.
    features : "common" (features every dataset has, including holdouts), "core", "all", or a list.
    splits : name -> (train_df, test_df) to override the default `temporal_split` for that dataset.
    holdout : name -> DataFrame used **only** as unseen test data (never trained on).
    bootstrap : bootstrap resamples for confidence intervals (0 = skip).
    unseen_on : "full" tests other datasets on all their rows; "test" uses only their test period, so
        every model scored on a dataset sees exactly the same rows (like-for-like comparisons and curves).
    model_factory : callable(features=..., **model_kwargs) returning an unfitted model (default:
        SymptomModel), e.g. an entry of `symptompred.registry()`.
    model_kwargs : passed to the model (e.g. n_models=5).

    Returns
    -------
    Long DataFrame, one row per (train, test) pair. `kind` is "within" (same dataset, later period)
    or "unseen" (a different dataset entirely). Use `pivot()` for a matrix.
    `.attrs["predictions"][(train, test)]` holds `(y_true, y_score)` for plotting curves.
    """
    from .model import SymptomModel

    datasets = {k: validate(v) for k, v in datasets.items()}
    holdout = {k: validate(v) for k, v in (holdout or {}).items()}
    if features == "common":
        features = common_features(*datasets.values(), *holdout.values(), candidates=ALL_FEATURES)
        if not features:
            raise ValueError("The datasets have no features in common.")
    splits = {name: (splits or {}).get(name) or temporal_split(df) for name, df in datasets.items()}

    rows, predictions = [], Predictions()
    for train_name, (train_df, own_test) in splits.items():
        model = (model_factory or SymptomModel)(features=features, **model_kwargs).fit(train_df)
        if unseen_on not in ("full", "test"):
            raise ValueError("unseen_on must be 'full' or 'test'")
        others = {n: (splits[n][1] if unseen_on == "test" else d) for n, d in datasets.items() if n != train_name}
        tests = {train_name: own_test, **others, **holdout}
        for test_name, test_df in tests.items():
            p = model.predict(test_df)
            predictions[(train_name, test_name)] = (test_df["y"].values, p)
            row = {"train": train_name, "test": test_name,
                   "kind": "within" if test_name == train_name else "unseen",
                   **metrics(test_df["y"].values, p)}
            if bootstrap:
                row.update(bootstrap_ci(test_df["y"].values, p, n=bootstrap))
            rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["features"] = list(features)
    result.attrs["predictions"] = predictions
    return result


def leave_one_out(datasets: dict[str, pd.DataFrame], splits: dict | None = None, features="all",
                  sample_weight="balance_sources", bootstrap: int = 0, model_factory=None,
                  **model_kwargs) -> pd.DataFrame:
    """For each dataset, pool every *other* dataset's training data, then test on this one's test period.

    This is the honest check of "would a pooled model work on a country it has never seen?". With only
    two datasets it reduces to "train on the other one".

    Parameters
    ----------
    datasets, splits, bootstrap, model_factory, model_kwargs : as for `compare`.
    features : "all" (union of the training datasets' features), "common" (only features every dataset
        has, including the held-out one), or a list.
    sample_weight : passed to `SymptomModel.fit` when pooling 2+ datasets (default: equal weight per source).

    `.attrs["predictions"][held_out]` holds `(y_true, y_score)` for plotting curves.
    """
    from .model import SymptomModel
    from .schema import pool

    datasets = {k: validate(v) for k, v in datasets.items()}
    splits = {name: (splits or {}).get(name) or temporal_split(df) for name, df in datasets.items()}
    rows, predictions = [], Predictions()
    for held_out in datasets:
        others = {n: splits[n][0] for n in datasets if n != held_out}
        feats = (common_features(*datasets.values()) if features == "common" else features)
        model = (model_factory or SymptomModel)(features=feats, **model_kwargs).fit(
            pool(others), sample_weight=sample_weight if len(others) > 1 else None)
        test_df = splits[held_out][1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the held-out country may lack some pooled features; expected
            p = model.predict(test_df)
        predictions[held_out] = (test_df["y"].values, p)
        row = {"held_out": held_out, "trained_on": " + ".join(others), "features": len(model.features_),
               **metrics(test_df["y"].values, p)}
        if bootstrap:
            row.update(bootstrap_ci(test_df["y"].values, p, n=bootstrap))
        rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["predictions"] = predictions
    return result


def forward_selection(train: pd.DataFrame, test: pd.DataFrame, candidates="all", selection_fraction: float = 0.25,
                      seed: int = 0, **model_kwargs) -> pd.DataFrame:
    """Add features one at a time, each time picking whichever helps most: "how many questions do we need?"

    The choice at each step is made on a held-out part of `train` (`selection_fraction`), never on `test`,
    so the test scores are honest. Each step's model is then refit on all of `train` and scored on `test`.

    Parameters
    ----------
    candidates : "all", "core", or a list of features to choose from (only those available in `train`).
    model_kwargs : passed to SymptomModel (e.g. n_models=3, missing="absent").

    Returns
    -------
    One row per step: the feature added, the features so far, and selection / test auROC.
    """
    from sklearn.model_selection import train_test_split

    from .model import SymptomModel
    from .schema import CORE_FEATURES, available_features

    train, test = validate(train), validate(test)
    pool_ = {"all": ALL_FEATURES, "core": CORE_FEATURES}.get(candidates, candidates) if isinstance(candidates, str) else candidates
    remaining = [f for f in available_features(train, tuple(pool_))]
    fit_part, sel_part = train_test_split(train, test_size=selection_fraction, random_state=seed, stratify=train["y"])
    chosen, rows = [], []
    while remaining:
        scores = {f: SymptomModel(features=chosen + [f], seed=seed, **model_kwargs).fit(fit_part)
                  .evaluate(sel_part, bootstrap=0)["auroc"] for f in remaining}
        best = max(scores, key=scores.get)
        chosen.append(best)
        remaining.remove(best)
        test_auroc = SymptomModel(features=list(chosen), seed=seed, **model_kwargs).fit(train).evaluate(test, bootstrap=0)["auroc"]
        rows.append({"step": len(chosen), "added": best, "features": list(chosen),
                     "selection_auroc": scores[best], "test_auroc": test_auroc})
    return pd.DataFrame(rows)


def pivot(results: pd.DataFrame, metric: str = "auroc") -> pd.DataFrame:
    """Matrix view of `compare()` output: rows = trained on, columns = tested on."""
    order = list(dict.fromkeys(results["train"]))
    order += [t for t in dict.fromkeys(results["test"]) if t not in order]
    return results.pivot(index="train", columns="test", values=metric).reindex(columns=order)


# ----------------------------------------------------------------------- headroom
def oracle_ceiling(df: pd.DataFrame, features) -> dict:
    """The best auROC / auPRC *any* model could score on `df` using only `features`.

    Every row with the same answers must get the same prediction, so the best possible ranking orders the
    distinct answer combinations by their actual positive rate *in this data*. This "model" is fitted on
    the test set itself, so it isn't a real model: it's the ceiling. Missing values count as their own answer.
    """
    df = validate(df)
    features = list(features)
    cell_rate = df.groupby(features, dropna=False)["y"].transform("mean").values
    m = metrics(df["y"].values, cell_rate)
    return {"auroc": m["auroc"], "auprc": m["auprc"],
            "combinations": int(df.groupby(features, dropna=False).ngroups)}


# ----------------------------------------------------------------- model selection
def cross_validate(model_factory, train: pd.DataFrame, folds: int = 5, seed: int = 0, sample_weight=None,
                   **model_kwargs) -> dict:
    """Stratified k-fold CV **within the training data**: mean auROC / auPRC and out-of-fold predictions."""
    from sklearn.model_selection import StratifiedKFold

    train = validate(train).reset_index(drop=True)
    oof = np.zeros(len(train))
    weights = None if sample_weight is None else np.asarray(sample_weight)
    for tr, va in StratifiedKFold(folds, shuffle=True, random_state=seed).split(train, train["y"]):
        model = model_factory(seed=seed, **model_kwargs)
        model.fit(train.iloc[tr], sample_weight=None if weights is None else weights[tr])
        oof[va] = model.predict(train.iloc[va])
    m = metrics(train["y"].values, oof)
    return {"cv_auroc": m["auroc"], "cv_auprc": m["auprc"], "oof": oof}


def select_and_score(candidates: dict, train: pd.DataFrame, test: pd.DataFrame, folds: int = 5, seed: int = 0,
                     metric: str = "cv_auroc", **model_kwargs) -> tuple[str, pd.DataFrame, object]:
    """Pick the best of `candidates` ({label: factory}) by CV on `train` only, refit it on all of `train`.

    The test data is only used by the caller, after this returns; the assertion guarantees the selection
    never sees a row from the test period. Returns (best label, CV table, fitted best model).
    """
    if not train["date"].max() < test["date"].min():
        raise AssertionError("Training data overlaps the test period: model selection would leak.")
    rows = {label: {k: v for k, v in cross_validate(factory, train, folds, seed, **model_kwargs).items() if k != "oof"}
            for label, factory in candidates.items()}
    table = pd.DataFrame(rows).T.sort_values(metric, ascending=False)
    best = table.index[0]
    return best, table, candidates[best](seed=seed, **model_kwargs).fit(train)


# ------------------------------------------------------------------------ pooling
POOL_VARIANTS = {
    "shared": "pooled, shared features",
    "all": "pooled, all features",
    "all+source": "pooled, all features + country",
}


def pool_vs_specialist(splits: dict[str, tuple[pd.DataFrame, pd.DataFrame]], model_factory=None,
                       variants=("shared", "all", "all+source"), sample_weight=None, bootstrap: int = 0,
                       **model_kwargs) -> pd.DataFrame:
    """For one technique: one model per country ("specialist") vs one model for all countries ("pooled").

    Each pooled model is trained on every country's training period combined (`pool`), then every model is
    scored on each country's test period.

    variants :
        "shared": only features every country has.
        "all": every feature any country has; NaN where a country never collected one.
        "all+source": as "all", plus a 0/1 column per country (`source_feature=True`).
    model_factory : callable(features=..., **model_kwargs) -> unfitted model (default: SymptomModel).
    sample_weight : for the pooled models, e.g. "balance_sources".

    Returns a long table (test, model, metrics); `.attrs["predictions"][(test, model)]` = (y, p).
    """
    from .model import SymptomModel
    from .schema import pool

    factory = model_factory or SymptomModel
    shared = common_features(*(d for pair in splits.values() for d in pair))
    specs = {"shared": dict(features=shared), "all": dict(features="all"),
             "all+source": dict(features="all", source_feature=True)}
    pooled_train = pool({name: tr for name, (tr, te) in splits.items()})
    models = {"specialist": {name: factory(features="all", **model_kwargs).fit(tr) for name, (tr, te) in splits.items()}}
    for v in variants:
        m = factory(**specs[v], **model_kwargs).fit(pooled_train, sample_weight=sample_weight)
        models[POOL_VARIANTS[v]] = {name: m for name in splits}

    rows, predictions = [], Predictions()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pooled models see features some countries never collected; expected
        for name, (tr, te) in splits.items():
            te = te.assign(source=name)
            for label, per_country in models.items():
                p = per_country[name].predict(te)
                predictions[(name, label)] = (te["y"].values, p)
                row = {"test": name, "model": label, **metrics(te["y"].values, p)}
                if bootstrap:
                    row.update(bootstrap_ci(te["y"].values, p, n=bootstrap))
                rows.append(row)
    result = pd.DataFrame(rows)
    result.attrs["predictions"] = predictions
    result.attrs["shared_features"] = shared
    return result

"""ROC and precision–recall curves for comparing models."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

# Fixed categorical order (validated palette); used when no colours are given
DEFAULT_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
REFERENCE = "#8a8984"


def plot_curves(curves: dict, axes=None, colors: dict | None = None, title: str | None = None,
                legend: bool = True, linestyles: dict | None = None):
    """Draw ROC (left) and precision–recall (right) curves for several models on one test set.

    Parameters
    ----------
    curves : {label: (y_true, y_score)}
    axes : (ax_roc, ax_pr) to draw into; a new 1x2 figure is created if omitted.
    colors, linestyles : {label: value}; colours default to a fixed categorical order.
    title : prefix for both panel titles (e.g. the test dataset's name).

    The ROC panel is plotted as sensitivity vs specificity, like the paper. The PR panel shows
    the test set's positive rate as a dashed line: that's what a model with no skill would score.
    """
    import matplotlib.pyplot as plt

    if axes is None:
        _, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax_roc, ax_pr = axes
    colors = colors or {}
    linestyles = linestyles or {}
    base_rate = None

    for i, (label, (y, p)) in enumerate(curves.items()):
        y, p = np.asarray(y), np.asarray(p)
        color = colors.get(label, DEFAULT_COLORS[i % len(DEFAULT_COLORS)])
        ls = linestyles.get(label, "-")
        fpr, tpr, _ = roc_curve(y, p)
        precision, recall, _ = precision_recall_curve(y, p)
        ax_roc.plot(1 - fpr, tpr, color=color, linewidth=2, linestyle=ls, label=f"{label} ({roc_auc_score(y, p):.3f})")
        ax_pr.plot(recall, precision, color=color, linewidth=2, linestyle=ls,
                   label=f"{label} ({average_precision_score(y, p):.3f})")
        base_rate = y.mean()

    ax_roc.plot([1, 0], [0, 1], color=REFERENCE, linestyle="--", linewidth=1)
    ax_roc.set(xlim=(1.01, -0.01), ylim=(-0.01, 1.01), xlabel="Specificity", ylabel="Sensitivity",
               title=f"{title}: ROC" if title else "ROC curve")
    if base_rate is not None:
        ax_pr.axhline(base_rate, color=REFERENCE, linestyle="--", linewidth=1)
        ax_pr.text(0.99, base_rate, f"positive rate {base_rate:.1%}", ha="right", va="bottom", fontsize=8, color="#52514e")
    ax_pr.set(xlim=(-0.01, 1.01), ylim=(-0.01, 1.01), xlabel="Sensitivity (recall)", ylabel="Precision (PPV)",
              title=f"{title}: precision–recall" if title else "Precision–recall curve")
    if legend:
        ax_roc.legend(frameon=False, fontsize=8, loc="lower left", title="auROC", title_fontsize=8)
        ax_pr.legend(frameon=False, fontsize=8, loc="upper right", title="auPRC", title_fontsize=8)
    return ax_roc, ax_pr


SHAP_COLORS = {1: "#e34948", 0: "#2a78d6", "missing": "#b5b4ad"}


def shap_feature_order(explanations: dict) -> list[str]:
    """Features across all explanations, most important first (mean |SHAP|, averaged over the datasets that have them)."""
    importance = {}
    for expl in explanations.values():
        for i, f in enumerate(expl.feature_names):
            importance.setdefault(f, []).append(np.abs(expl.values[:, i]).mean())
    return sorted(importance, key=lambda f: -np.mean(importance[f]))


def plot_shap_grid(explanations: dict, order: list[str] | None = None, max_points: int = 4000,
                   sharex: bool = False, seed: int = 0, panel_width: float = 4.6, suptitle: str | None = None):
    """Side-by-side SHAP beeswarms with the same features, in the same order, on every panel.

    explanations : {name: shap.Explanation} (e.g. from `SymptomModel.shap_values`)
    order : feature order, top to bottom; defaults to `shap_feature_order(explanations)`.
        Features a dataset doesn't have are shown as an empty row marked "not collected".
    max_points : random sample of rows per panel, to keep plotting fast (the pattern is unchanged).
    sharex : use one SHAP scale for every panel (shows which models have bigger effects overall).

    Points are coloured by the feature's value: 1 (red), 0 (blue) or missing (grey), with vertical
    jitter so the many identical values of a binary feature spread into a visible band.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    order = order or shap_feature_order(explanations)
    rng = np.random.default_rng(seed)
    n = len(explanations)
    fig, axes = plt.subplots(1, n, figsize=(panel_width * n, 0.42 * len(order) + 1.6), sharey=True,
                             sharex=sharex, squeeze=False)
    for ax, (name, expl) in zip(axes[0], explanations.items()):
        rows = rng.choice(len(expl.values), size=min(max_points, len(expl.values)), replace=False)
        for y, f in enumerate(order):
            if f not in expl.feature_names:
                ax.text(0.5, y, "not collected", transform=ax.get_yaxis_transform(), ha="center", va="center",
                        fontsize=8, color="#8a8984", style="italic")
                continue
            i = expl.feature_names.index(f)
            v, x = expl.values[rows, i], expl.data[rows, i]
            for key, mask in [("missing", np.isnan(x)), (0, x == 0), (1, x == 1)]:
                if mask.any():
                    ax.scatter(v[mask], y + rng.uniform(-0.28, 0.28, mask.sum()), s=4, alpha=0.35,
                               color=SHAP_COLORS[key], linewidths=0, rasterized=True)
        ax.axvline(0, color="#8a8984", linewidth=1)
        ax.set_title(name)
        ax.set_xlabel("SHAP value (log-odds)")
        ax.grid(axis="y", visible=False)
    axes[0, 0].set_yticks(range(len(order)), order)
    axes[0, 0].set_ylim(len(order) - 0.5, -0.5)
    handles = [Line2D([], [], marker="o", linestyle="", color=c, label=l)
               for l, c in [("feature = 1", SHAP_COLORS[1]), ("feature = 0", SHAP_COLORS[0]), ("missing", SHAP_COLORS["missing"])]]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def plot_curve_grid(panels: dict, colors: dict | None = None, linestyles: dict | None = None,
                    suptitle: str | None = None, panel_width: float = 4.2):
    """One column per test set, ROC on top and precision–recall below.

    panels : {test_name: {label: (y_true, y_score)}}
    """
    import matplotlib.pyplot as plt

    n = len(panels)
    fig, axes = plt.subplots(2, n, figsize=(panel_width * n, 8.4), squeeze=False)
    for j, (test_name, curves) in enumerate(panels.items()):
        plot_curves(curves, axes=(axes[0, j], axes[1, j]), colors=colors, linestyles=linestyles, title=test_name)
    if suptitle:
        fig.suptitle(suptitle, fontweight="bold")
    fig.tight_layout()
    return fig

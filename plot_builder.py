"""Plot construction functions. Each returns a matplotlib Figure.

All functions share the signature (df, **options) -> plt.Figure so the
Streamlit app can dispatch on plot type generically. Validation errors are
raised as PlotError with a message safe to show directly to the user.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from style import finalize_axes


class PlotError(ValueError):
    pass


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise PlotError(message)


def _new_fig(figsize):
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax


def scatter(df, x, y, hue=None, size=None, style=None, marker="o", marker_map=None,
            marker_size=None, show_outline=True, outline_color="#FFFFFF", outline_width=0.3,
            palette=None, figsize=(4, 3.2), add_regression=False, point_alpha=0.85,
            title="", xlabel="", ylabel=""):
    _require(x and y, "Choose both an X and a Y column.")
    fig, ax = _new_fig(figsize)
    scatter_kwargs = dict(
        data=df, x=x, y=y, hue=hue, size=size, palette=palette,
        alpha=point_alpha, ax=ax,
        edgecolor=outline_color if show_outline else "none",
        linewidth=outline_width if show_outline else 0,
    )
    if marker_size is not None and size is None:
        scatter_kwargs["s"] = marker_size
    if style:
        scatter_kwargs["style"] = style
        if marker_map:
            scatter_kwargs["markers"] = marker_map
    else:
        scatter_kwargs["marker"] = marker
    sns.scatterplot(**scatter_kwargs)
    if add_regression:
        sns.regplot(
            data=df, x=x, y=y, scatter=False, ax=ax,
            color="black", line_kws={"linewidth": 1.0, "linestyle": "--"},
        )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def line(df, x, y, hue=None, palette=None, figsize=(4, 3.2), show_markers=True,
         marker="o", marker_map=None, ci=None, title="", xlabel="", ylabel=""):
    _require(x and y, "Choose both an X and a Y column.")
    fig, ax = _new_fig(figsize)
    line_kwargs = dict(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, errorbar=ci)
    if show_markers:
        if hue and marker_map:
            line_kwargs["style"] = hue
            line_kwargs["markers"] = marker_map
            line_kwargs["dashes"] = False
        else:
            line_kwargs["marker"] = marker
    sns.lineplot(**line_kwargs)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def bar(df, x, y, hue=None, palette=None, figsize=(4, 3.2), estimator="mean",
        error_bars="sd", bar_label_mode="none", bar_label_fmt="%.2f",
        xtick_rotation=0, y_min=None, y_max=None, title="", xlabel="", ylabel=""):
    _require(x and y, "Choose both an X (category) and a Y (numeric) column.")
    if bar_label_mode == "category":
        _require(hue is not None, "Category-on-bar labels need a Hue column to be set.")
    if y_min is not None and y_max is not None:
        _require(y_min < y_max, "Y min must be less than Y max.")
    fig, ax = _new_fig(figsize)
    errorbar = None if error_bars == "none" else error_bars
    sns.barplot(
        data=df, x=x, y=y, hue=hue, palette=palette, ax=ax,
        estimator=estimator, errorbar=errorbar, capsize=0.12,
        err_kws={"linewidth": 0.8}, edgecolor="black", linewidth=0.6,
    )

    if bar_label_mode == "value":
        for container in ax.containers:
            ax.bar_label(container, fmt=bar_label_fmt, padding=3)
        ax.margins(y=0.12)
    elif bar_label_mode == "category":
        for container in ax.containers:
            label = container.get_label()
            ax.bar_label(container, labels=[label] * len(container), rotation=90, padding=3)
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        ax.margins(y=0.25)

    if xtick_rotation:
        ha = "right" if 0 < xtick_rotation < 90 else "center"
        for label in ax.get_xticklabels():
            label.set_rotation(xtick_rotation)
            label.set_ha(ha)

    if y_min is not None or y_max is not None:
        ax.set_ylim(bottom=y_min, top=y_max)

    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def box(df, x, y, hue=None, palette=None, figsize=(4, 3.2), show_points=True,
        title="", xlabel="", ylabel=""):
    _require(x and y, "Choose both an X (category) and a Y (numeric) column.")
    fig, ax = _new_fig(figsize)
    sns.boxplot(
        data=df, x=x, y=y, hue=hue, palette=palette, ax=ax,
        fliersize=0 if show_points else 2.5, linewidth=0.8, width=0.6,
    )
    if show_points:
        sns.stripplot(
            data=df, x=x, y=y, hue=hue, ax=ax, dodge=bool(hue),
            size=2.5, color="black", alpha=0.5, legend=False,
        )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def violin(df, x, y, hue=None, palette=None, figsize=(4, 3.2), inner="box",
           title="", xlabel="", ylabel=""):
    _require(x and y, "Choose both an X (category) and a Y (numeric) column.")
    fig, ax = _new_fig(figsize)
    sns.violinplot(
        data=df, x=x, y=y, hue=hue, palette=palette, ax=ax,
        inner=inner, linewidth=0.8, cut=0,
    )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def histogram(df, x, hue=None, palette=None, figsize=(4, 3.2), bins=30,
              kde=True, stat="count", title="", xlabel="", ylabel=""):
    _require(x, "Choose a column to plot the distribution of.")
    fig, ax = _new_fig(figsize)
    sns.histplot(
        data=df, x=x, hue=hue, palette=palette, ax=ax, bins=bins,
        kde=kde, stat=stat, edgecolor="white", linewidth=0.3, alpha=0.85,
    )
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or stat.capitalize())
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def kde(df, x, y=None, hue=None, palette=None, figsize=(4, 3.2), fill=True,
        title="", xlabel="", ylabel=""):
    _require(x, "Choose at least an X column.")
    fig, ax = _new_fig(figsize)
    if y:
        sns.kdeplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax, fill=fill, thresh=0.05)
    else:
        sns.kdeplot(data=df, x=x, hue=hue, palette=palette, ax=ax, fill=fill, linewidth=1.0)
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or (y or "Density"))
    if title:
        ax.set_title(title)
    finalize_axes(ax)
    fig.tight_layout()
    return fig


def heatmap(df, columns=None, figsize=(4.5, 4), cmap="rocket", annotate=True,
            value_decimals=2, method="pearson", title="", xlabel="", ylabel=""):
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    _require(len(cols) >= 2, "Select at least two numeric columns for a correlation heatmap.")
    corr = df[cols].corr(method=method)
    fig, ax = _new_fig(figsize)
    sns.heatmap(
        corr, ax=ax, cmap=cmap, annot=annotate, fmt=f".{value_decimals}f",
        annot_kws={"size": 6}, linewidths=0.4, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": f"{method.capitalize()} r"},
        square=True, vmin=-1, vmax=1,
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _dedupe_labels(labels):
    counts = {}
    out = []
    for v in labels:
        counts[v] = counts.get(v, 0) + 1
        out.append(v if counts[v] == 1 else f"{v} (#{counts[v]})")
    return out


def settings_heatmap(df, metric_columns, label_columns=None, normalize="minmax",
                      invert_columns=None, annotate=True, value_fmt=".3f", cmap="rocket",
                      figsize=(5, 4), title="", xlabel="", ylabel=""):
    _require(metric_columns and len(metric_columns) >= 1, "Select at least one metric column.")
    invert_columns = set(invert_columns or [])

    if label_columns:
        labels = df[label_columns].astype(str).agg(" | ".join, axis=1).tolist()
    else:
        labels = df.index.astype(str).tolist()
    labels = _dedupe_labels(labels)

    raw = df[metric_columns].astype(float)
    raw.index = labels

    if normalize == "none":
        color_matrix = raw.copy()
        cbar_label = "Value"
    elif normalize == "minmax":
        span = raw.max() - raw.min()
        color_matrix = (raw - raw.min()) / span.replace(0, 1)
        cbar_label = "Min-max scaled (per column)"
    elif normalize == "zscore":
        std = raw.std().replace(0, 1)
        color_matrix = (raw - raw.mean()) / std
        cbar_label = "Z-score (per column)"
    else:
        raise PlotError(f"Unknown normalization: {normalize}")

    for col in invert_columns:
        if col in color_matrix.columns:
            color_matrix[col] = -color_matrix[col] if normalize == "zscore" else (1 - color_matrix[col])

    fig, ax = _new_fig(figsize)
    annot_data = raw.map(lambda v: format(v, value_fmt)) if annotate else False
    sns.heatmap(
        color_matrix, ax=ax, cmap=cmap, annot=annot_data, fmt="",
        annot_kws={"size": 7}, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": cbar_label},
    )
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel or ("Setting" if label_columns else ""))
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def pairplot(df, columns=None, hue=None, palette=None, height=1.6, diag_kind="kde",
             title=""):
    cols = columns or df.select_dtypes(include=np.number).columns.tolist()
    _require(len(cols) >= 2, "Select at least two numeric columns for a pairplot.")
    use_cols = cols + ([hue] if hue and hue not in cols else [])
    grid = sns.pairplot(
        df[use_cols], vars=cols, hue=hue, palette=palette,
        height=height, diag_kind=diag_kind,
        plot_kws={"alpha": 0.75, "s": 15, "edgecolor": "white", "linewidth": 0.2},
    )
    for ax in grid.axes.flat:
        if ax is not None:
            sns.despine(ax=ax)
    if title:
        grid.fig.suptitle(title, y=1.02)
    return grid.fig


PLOT_REGISTRY = {
    "Scatter": {
        "func": scatter,
        "needs": ["x", "y"],
        "optional": ["hue", "size", "style"],
        "description": "Relationship between two numeric variables; optional regression line.",
    },
    "Line": {
        "func": line,
        "needs": ["x", "y"],
        "optional": ["hue"],
        "description": "Trend over an ordered/continuous X (e.g. time series).",
    },
    "Bar": {
        "func": bar,
        "needs": ["x", "y"],
        "optional": ["hue"],
        "description": "Summary statistic (mean/median) per category with error bars.",
    },
    "Box": {
        "func": box,
        "needs": ["x", "y"],
        "optional": ["hue"],
        "description": "Distribution per category: median, quartiles, outliers.",
    },
    "Violin": {
        "func": violin,
        "needs": ["x", "y"],
        "optional": ["hue"],
        "description": "Full distribution shape per category.",
    },
    "Histogram": {
        "func": histogram,
        "needs": ["x"],
        "optional": ["hue"],
        "description": "Frequency distribution of one numeric variable.",
    },
    "Density (KDE)": {
        "func": kde,
        "needs": ["x"],
        "optional": ["y", "hue"],
        "description": "Smoothed density; 1D or 2D (contour) if Y is also set.",
    },
    "Correlation heatmap": {
        "func": heatmap,
        "needs": ["columns"],
        "optional": [],
        "description": "Pairwise correlation matrix across selected numeric columns.",
    },
    "Pairplot": {
        "func": pairplot,
        "needs": ["columns"],
        "optional": ["hue"],
        "description": "Grid of pairwise scatter/histograms across numeric columns.",
    },
    "Settings heatmap": {
        "func": settings_heatmap,
        "needs": ["metric_columns"],
        "optional": ["label_columns"],
        "description": "Compare metrics (e.g. AUC, RMSE) across settings/configs, scaled per column.",
    },
}

"""Plot construction functions. Each returns a matplotlib Figure.

All functions share the signature (df, **options) -> plt.Figure so the
Streamlit app can dispatch on plot type generically. Validation errors are
raised as PlotError with a message safe to show directly to the user.
"""

from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

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


def _container_hue_labels(ax):
    """Map each ax.containers entry to its real hue-level label.

    container.get_label() returns a placeholder like "_container0" in
    current seaborn — the actual level names only live in the legend's
    handle/label pairs, which share the same order as ax.containers.
    """
    _, labels = ax.get_legend_handles_labels()
    if labels and len(labels) == len(ax.containers):
        return labels
    return [c.get_label() for c in ax.containers]


def bar(df, x, y, hue=None, palette=None, figsize=(4, 3.2), estimator="mean",
        error_bars="sd", bar_label_mode="none", label_decimals=2,
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
            ax.bar_label(container, fmt=f"%.{label_decimals}f", padding=3)
        ax.margins(y=0.12)
    elif bar_label_mode == "mean_std":
        dispersion = "se" if error_bars == "se" else "sd"
        x_order = [t.get_text() for t in ax.get_xticklabels()]

        def _mean_disp(s):
            m = s.mean()
            d = s.std(ddof=1) if len(s) > 1 else 0.0
            if dispersion == "se" and len(s) > 1:
                d = d / np.sqrt(len(s))
            return m, d

        def _annotate(x_center, m, d):
            y_top = m + d if error_bars != "none" else m
            ax.text(
                x_center, y_top, f"{m:.{label_decimals}f} ± {d:.{label_decimals}f}",
                ha="center", va="bottom",
            )

        if hue:
            # Normal grouped-bar layout: one container per hue level, patches in x order.
            stats = {(str(xv), str(hv)): _mean_disp(sub) for (xv, hv), sub in df.groupby([x, hue])[y]}
            for container, level in zip(ax.containers, _container_hue_labels(ax)):
                for i, patch in enumerate(container.patches):
                    if i >= len(x_order):
                        continue
                    key = (x_order[i], str(level))
                    if key in stats:
                        _annotate(patch.get_x() + patch.get_width() / 2, *stats[key])
        else:
            # Without hue, seaborn may split bars into one single-patch container per
            # category (when a palette is set) instead of one container holding all of
            # them — match by actual x-position rather than trusting container layout.
            stats = {str(xv): _mean_disp(sub) for xv, sub in df.groupby(x)[y]}
            for patch in (p for c in ax.containers for p in c.patches):
                x_center = patch.get_x() + patch.get_width() / 2
                idx = round(x_center)
                if 0 <= idx < len(x_order) and x_order[idx] in stats:
                    _annotate(x_center, *stats[x_order[idx]])
        ax.margins(y=0.2)
    elif bar_label_mode == "category":
        for container, label in zip(ax.containers, _container_hue_labels(ax)):
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


def pivot_heatmap(df, x, y, value, annotate=True, value_decimals=3, cmap="rocket",
                   highlight_best=True, figsize=(4.5, 4), title="", xlabel="", ylabel=""):
    _require(x and y and value, "Choose X, Y, and a Value column.")
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc="mean")
    fig, ax = _new_fig(figsize)
    sns.heatmap(
        pivot, ax=ax, cmap=cmap, annot=annotate, fmt=f".{value_decimals}f",
        annot_kws={"size": 7}, linewidths=0.5, linecolor="white",
        cbar_kws={"shrink": 0.8, "label": value},
    )
    if highlight_best and not np.all(np.isnan(pivot.values)):
        r, c = np.unravel_index(np.nanargmax(pivot.values), pivot.values.shape)
        ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=False, edgecolor="black", linewidth=2))
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel or y)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def _holm_correct(pvals):
    """Holm-Bonferroni step-down correction for a 1D array of p-values."""
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(order):
        running_max = max(running_max, (m - rank) * pvals[idx])
        adjusted[idx] = min(running_max, 1.0)
    return adjusted


def model_significance(df, group, value, pair_id=None, nonparametric=False, correction="holm",
                        alpha=0.05, annotate=True, value_decimals=3, cmap="rocket",
                        figsize=(4.5, 4), title="", xlabel="", ylabel=""):
    """Pairwise statistical comparison between groups (e.g. models) on one metric.

    If `pair_id` is given, each pair of groups is compared on matching pair_id
    rows (e.g. same dataset/fold) with a paired test (paired t / Wilcoxon).
    Otherwise an independent-samples test is used (Welch's t / Mann-Whitney U).
    """
    _require(group and value, "Choose a Model/Group column and a Value column.")
    groups = sorted(df[group].dropna().unique().tolist(), key=str)
    _require(len(groups) >= 2, "Need at least two models/groups to compare.")

    pairs = list(combinations(groups, 2))
    raw_p = []
    for g1, g2 in pairs:
        s1 = df.loc[df[group] == g1]
        s2 = df.loc[df[group] == g2]
        if pair_id:
            merged = s1.set_index(pair_id)[value].rename("a").to_frame().join(
                s2.set_index(pair_id)[value].rename("b"), how="inner"
            )
            _require(len(merged) >= 2, f"Not enough matching '{pair_id}' rows for {g1} vs {g2}.")
            a, b = merged["a"].to_numpy(), merged["b"].to_numpy()
            _, p = stats.wilcoxon(a, b) if nonparametric else stats.ttest_rel(a, b)
        else:
            a, b = s1[value].to_numpy(), s2[value].to_numpy()
            _require(len(a) >= 2 and len(b) >= 2, f"Not enough observations for {g1} vs {g2}.")
            _, p = stats.mannwhitneyu(a, b) if nonparametric else stats.ttest_ind(a, b, equal_var=False)
        raw_p.append(p)

    raw_p = np.array(raw_p)
    if correction == "bonferroni":
        adj_p = np.clip(raw_p * len(raw_p), 0, 1)
    elif correction == "holm":
        adj_p = _holm_correct(raw_p)
    else:
        adj_p = raw_p

    n = len(groups)
    pmat = pd.DataFrame(np.nan, index=groups, columns=groups)
    for (g1, g2), p in zip(pairs, adj_p):
        pmat.loc[g1, g2] = p
        pmat.loc[g2, g1] = p

    def _stars(p):
        if pd.isna(p):
            return ""
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < alpha:
            return "*"
        return "ns"

    fig, ax = _new_fig(figsize)
    annot = pmat.map(lambda p: "" if pd.isna(p) else f"{p:.{value_decimals}f}\n{_stars(p)}") if annotate else False
    cbar_label = "p-value" + (f" ({correction})" if correction != "none" else "")
    sns.heatmap(
        pmat, ax=ax, mask=np.eye(n, dtype=bool), cmap=cmap, annot=annot, fmt="",
        annot_kws={"size": 6.5}, linewidths=0.5, linecolor="white",
        vmin=0, vmax=1, cbar_kws={"shrink": 0.8, "label": cbar_label},
    )
    for label in ax.get_xticklabels():
        label.set_rotation(30)
        label.set_ha("right")
    ax.set_xlabel(xlabel or group)
    ax.set_ylabel(ylabel or group)
    if title:
        ax.set_title(title)
    fig.tight_layout()
    return fig


def critical_difference_diagram(df, group, value, pair_id, higher_is_better=True,
                                 alpha=0.05, figsize=(6, 3.5), title="", xlabel="", ylabel=""):
    """Classic Demsar-style CD diagram: average ranks of models across paired items
    (e.g. datasets), with a Friedman test and Nemenyi-corrected critical difference
    grouping models that are NOT significantly different (thick bars).
    """
    _require(
        group and value and pair_id,
        "Choose a Model/Group column, a Value column, and a pairing ID column "
        "(e.g. Dataset) — a CD diagram needs every model scored on the same items.",
    )
    wide = df.pivot_table(index=pair_id, columns=group, values=value, aggfunc="mean").dropna(axis=0, how="any")
    k = wide.shape[1]
    n = wide.shape[0]
    _require(k >= 3, "Need at least three models/groups to draw a critical difference diagram.")
    _require(n >= 2, f"Need at least two '{pair_id}' items with complete scores for every model.")

    ranks = wide.rank(axis=1, ascending=not higher_is_better)
    avg_ranks = ranks.mean(axis=0).sort_values()
    names = avg_ranks.index.astype(str).tolist()
    rank_vals = avg_ranks.to_numpy()

    stat, p_friedman = stats.friedmanchisquare(*[wide[c].to_numpy() for c in wide.columns])
    q_alpha = stats.studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    cd = q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))

    # Maximal cliques: runs of adjacent (rank-sorted) models within CD of each other.
    cliques = []
    i = 0
    while i < k:
        j = i
        while j + 1 < k and rank_vals[j + 1] - rank_vals[i] <= cd:
            j += 1
        if j > i:
            cliques.append((i, j))
        i += 1
    cliques = [c for c in cliques if not any(c != c2 and c2[0] <= c[0] and c[1] <= c2[1] for c2 in cliques)]

    # Classic layout: rank axis near the top; models split into a left column
    # (better half) and a right column (worse half), each its own vertical
    # stack of labels, so crowded/near-identical ranks never collide — only
    # the leader line's bend point depends on the exact rank value.
    lo, hi = 1, k
    half = (k + 1) // 2
    left_idx, right_idx = list(range(half)), list(range(half, k))

    y_cd = 0.55
    y_axis = 0.0
    y_ticknum = -0.16
    clique_top, clique_step = -0.36, -0.17
    n_clique_rows = max(len(cliques), 1)
    clique_bottom = clique_top + clique_step * (n_clique_rows - 1)
    label_top = clique_bottom - 0.4
    label_step = -0.44
    n_label_rows = max(len(left_idx), len(right_idx), 1)
    label_bottom = label_top + label_step * (n_label_rows - 1)

    y_top_lim = y_cd + 0.3
    y_bottom_lim = label_bottom - 0.3

    max_name_len = max((len(nm) for nm in names), default=8)
    gap_in = 0.15
    text_w_in = 0.072 * max_name_len + 0.1
    core_w_in = max(figsize[0] - 2.4, 0.8 * (k - 1) + 1.2)
    rank_to_in = core_w_in / (hi - lo)
    gap_rank = gap_in / rank_to_in
    text_rank = text_w_in / rank_to_in

    fig_w_in = (core_w_in + 2 * (gap_in + text_w_in)) / 0.96 + 0.1
    fig_h_in = (y_top_lim - y_bottom_lim) * 1.15
    fig, ax = _new_fig((fig_w_in, max(figsize[1], fig_h_in)))
    ax.set_position((0.01, 0.06, 0.98, 0.80))

    left_margin_x = lo - gap_rank
    right_margin_x = hi + gap_rank
    ax.set_xlim(left_margin_x - text_rank, right_margin_x + text_rank)
    ax.set_ylim(y_bottom_lim, y_top_lim)

    ax.plot([lo, hi], [y_axis, y_axis], color="black", linewidth=1.1)
    for r in range(lo, hi + 1):
        ax.plot([r, r], [y_axis - 0.06, y_axis + 0.06], color="black", linewidth=1.0)
        ax.text(r, y_ticknum, str(r), ha="center", va="top", fontsize=7)

    def _leader(name, r, row, side):
        row_y = label_top + row * label_step
        margin_x = left_margin_x if side == "left" else right_margin_x
        ax.plot([r, r], [y_axis, row_y], color="#999999", linewidth=0.7)
        ax.plot([r, margin_x], [row_y, row_y], color="#999999", linewidth=0.7)
        ax.text(
            margin_x, row_y, name, fontsize=7.5, va="center",
            ha="right" if side == "left" else "left",
        )

    for row, idx in enumerate(left_idx):
        _leader(names[idx], rank_vals[idx], row, "left")
    for row, idx in enumerate(right_idx):
        _leader(names[idx], rank_vals[idx], row, "right")

    y_c = clique_top
    for (i0, j0) in cliques:
        ax.plot([rank_vals[i0], rank_vals[j0]], [y_c, y_c], color="black", linewidth=3.5, solid_capstyle="butt")
        y_c += clique_step

    ax.plot([lo, lo + cd], [y_cd, y_cd], color="black", linewidth=1.5)
    ax.plot([lo, lo], [y_cd - 0.05, y_cd + 0.05], color="black", linewidth=1.5)
    ax.plot([lo + cd, lo + cd], [y_cd - 0.05, y_cd + 0.05], color="black", linewidth=1.5)
    ax.text(lo + cd / 2, y_cd + 0.08, f"CD = {cd:.3f}", ha="center", va="bottom", fontsize=7.5)

    ax.axis("off")
    ttl = title or "Critical Difference Diagram"
    subtitle = f"Friedman χ²={stat:.2f}, p={p_friedman:.4g}  (n={n} items, k={k} models, α={alpha})"
    ax.set_title(f"{ttl}\n{subtitle}", fontsize=8.5)
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
    "Pivot heatmap": {
        "func": pivot_heatmap,
        "needs": ["x", "y", "value"],
        "optional": [],
        "description": "Grid of one numeric value across two categorical variables — e.g. find the best combination of two settings.",
    },
    "Model comparison (stats)": {
        "func": model_significance,
        "needs": ["group", "value"],
        "optional": ["pair_id"],
        "description": "Pairwise statistical significance between models/groups — p-value matrix with significance stars.",
    },
    "Critical Difference Diagram": {
        "func": critical_difference_diagram,
        "needs": ["group", "value", "pair_id"],
        "optional": [],
        "description": "Average ranks across datasets with Friedman + Nemenyi test — groups models that are not significantly different.",
    },
}

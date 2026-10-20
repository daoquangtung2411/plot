import io

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from codegen import generate_script
from plot_builder import PLOT_REGISTRY, PlotError
from style import FIGURE_WIDTHS_MM, MARKER_SHAPES, OKABE_ITO, PALETTES, apply_nature_style, mm_to_inch
from utils import load_file, make_sample_data

st.set_page_config(page_title="Plot Studio", layout="wide")

NONE_LABEL = "— none —"
HEATMAP_CMAPS = ["rocket", "mako", "viridis", "coolwarm", "vlag", "Blues", "Greys"]


def col_options(df, kind="any"):
    if kind == "numeric":
        cols = df.select_dtypes(include="number").columns.tolist()
    elif kind == "categorical":
        cols = df.select_dtypes(exclude="number").columns.tolist()
        if not cols:
            cols = df.columns.tolist()
    else:
        cols = df.columns.tolist()
    return cols


def optional_selectbox(label, options, key):
    choice = st.selectbox(label, [NONE_LABEL] + options, key=key)
    return None if choice == NONE_LABEL else choice


# ---------------------------------------------------------------- data load
st.sidebar.title("1. Data")
source = st.sidebar.radio("Source", ["Upload file", "Use sample dataset"], label_visibility="collapsed")

df = None
if source == "Upload file":
    uploaded = st.sidebar.file_uploader("CSV / TSV / Excel", type=["csv", "tsv", "txt", "xlsx", "xls"])
    if uploaded is not None:
        try:
            df = load_file(uploaded)
        except Exception as e:
            st.sidebar.error(f"Could not read file: {e}")
else:
    df = make_sample_data()
    st.sidebar.caption("Synthetic lab dataset: group × timepoint × replicate measurements.")

if df is None:
    st.title("Plot Studio")
    st.info("Upload a CSV/TSV/Excel file, or switch to the sample dataset in the sidebar, to get started.")
    st.stop()

# ---------------------------------------------------------- plot type + cols
st.sidebar.title("2. Plot type")
plot_name = st.sidebar.selectbox("Type", list(PLOT_REGISTRY.keys()))
spec = PLOT_REGISTRY[plot_name]
st.sidebar.caption(spec["description"])

st.sidebar.title("3. Variables")
params = {}
all_cols = col_options(df)
numeric_cols = col_options(df, "numeric")
categorical_cols = col_options(df, "categorical")

for field in spec["needs"] + spec["optional"]:
    required = field in spec["needs"]
    label = f"{field} {'*' if required else '(optional)'}"
    if field == "columns":
        default = numeric_cols[: min(6, len(numeric_cols))]
        params["columns"] = st.sidebar.multiselect(
            "Numeric columns *", numeric_cols, default=default, key="cols_multi"
        )
    elif field == "y" and plot_name != "Density (KDE)":
        opts = numeric_cols or all_cols
        params["y"] = st.sidebar.selectbox(label, opts, key="field_y")
    elif field in ("x", "y"):
        opts = all_cols if not required else (numeric_cols or all_cols) if field == "y" else all_cols
        if required:
            params[field] = st.sidebar.selectbox(label, opts, key=f"field_{field}")
        else:
            params[field] = optional_selectbox(label, opts, key=f"field_{field}")
    elif field in ("hue", "style"):
        params[field] = optional_selectbox(label, all_cols, key=f"field_{field}")
    elif field == "size":
        params["size"] = optional_selectbox(label, numeric_cols, key="field_size")
    elif field == "metric_columns":
        float_cols = [c for c in numeric_cols if pd.api.types.is_float_dtype(df[c])]
        default = float_cols if float_cols else numeric_cols[: min(4, len(numeric_cols))]
        params["metric_columns"] = st.sidebar.multiselect(
            "Metric columns (numeric) *", numeric_cols, default=default, key="metric_cols_multi"
        )
    elif field == "label_columns":
        other_cols = [c for c in all_cols if c not in params.get("metric_columns", [])]
        params["label_columns"] = st.sidebar.multiselect(
            "Label columns (optional, build row name)", all_cols, default=other_cols, key="label_cols_multi"
        )

# ---------------------------------------------------------------- plot opts
st.sidebar.title("4. Plot options")
plot_kwargs = {}
if plot_name == "Scatter":
    plot_kwargs["add_regression"] = st.sidebar.checkbox("Add regression line", value=False)
    plot_kwargs["point_alpha"] = st.sidebar.slider("Point opacity", 0.1, 1.0, 0.85)

    shape_names = list(MARKER_SHAPES.keys())
    style_col = params.get("style")
    if style_col:
        categories = sorted(df[style_col].dropna().unique().tolist(), key=str)
        st.sidebar.markdown("**Marker shape per category**")
        marker_map = {}
        for i, cat in enumerate(categories):
            default_idx = i % len(shape_names)
            if i < 10:
                choice = st.sidebar.selectbox(
                    f"Shape: {cat}", shape_names, index=default_idx, key=f"shape_{style_col}_{cat}"
                )
                marker_map[cat] = MARKER_SHAPES[choice]
            else:
                marker_map[cat] = MARKER_SHAPES[shape_names[default_idx]]
        if len(categories) > 10:
            st.sidebar.caption(f"{len(categories) - 10} additional categories auto-assigned a shape.")
        plot_kwargs["marker_map"] = marker_map
    else:
        shape_choice = st.sidebar.selectbox("Marker shape", shape_names, index=0)
        plot_kwargs["marker"] = MARKER_SHAPES[shape_choice]

    if params.get("size"):
        plot_kwargs["marker_size"] = None
        st.sidebar.caption("Marker size is driven by the selected 'size' column.")
    else:
        plot_kwargs["marker_size"] = st.sidebar.slider("Marker size", 5, 300, 36, 1)

    st.sidebar.markdown("**Marker outline**")
    show_outline = st.sidebar.checkbox("Show outline", value=True)
    plot_kwargs["show_outline"] = show_outline
    if show_outline:
        plot_kwargs["outline_color"] = st.sidebar.color_picker("Outline color", "#FFFFFF")
        plot_kwargs["outline_width"] = st.sidebar.slider("Outline width", 0.1, 2.0, 0.3, 0.1)
    else:
        plot_kwargs["outline_color"] = "#FFFFFF"
        plot_kwargs["outline_width"] = 0.0
elif plot_name == "Line":
    plot_kwargs["show_markers"] = st.sidebar.checkbox("Show markers", value=True)
    ci_choice = st.sidebar.selectbox("Error band", ["None", "sd", "se"], index=0)
    plot_kwargs["ci"] = None if ci_choice == "None" else ci_choice

    hue_col = params.get("hue")
    shape_names = list(MARKER_SHAPES.keys())
    customize = False
    if hue_col:
        customize = st.sidebar.checkbox("Customize color & marker per line", value=False)

    if customize and hue_col:
        categories = sorted(df[hue_col].dropna().unique().tolist(), key=str)
        st.sidebar.markdown("**Per-line color & shape**")
        color_map, marker_map = {}, {}
        for i, cat in enumerate(categories):
            st.sidebar.caption(str(cat))
            c1, c2 = st.sidebar.columns(2)
            default_color = OKABE_ITO[i % len(OKABE_ITO)]
            color_map[cat] = c1.color_picker(
                "Color", default_color, key=f"line_color_{hue_col}_{cat}", label_visibility="collapsed"
            )
            default_shape_idx = i % len(shape_names)
            shape_choice = c2.selectbox(
                "Shape", shape_names, index=default_shape_idx,
                key=f"line_shape_{hue_col}_{cat}", label_visibility="collapsed",
            )
            marker_map[cat] = MARKER_SHAPES[shape_choice]
        plot_kwargs["color_map"] = color_map
        plot_kwargs["marker_map"] = marker_map if plot_kwargs["show_markers"] else None
    else:
        plot_kwargs["marker_map"] = None
        if plot_kwargs["show_markers"]:
            shape_choice = st.sidebar.selectbox("Marker shape", shape_names, index=0)
            plot_kwargs["marker"] = MARKER_SHAPES[shape_choice]
elif plot_name == "Bar":
    plot_kwargs["estimator"] = st.sidebar.selectbox("Summary statistic", ["mean", "median"])
    plot_kwargs["error_bars"] = st.sidebar.selectbox("Error bars", ["sd", "se", "none"])
    label_options = ["None", "Value on bar"]
    if params.get("hue"):
        label_options.append("Category on bar (hides legend)")
    label_map = {
        "None": "none",
        "Value on bar": "value",
        "Category on bar (hides legend)": "category",
    }
    label_choice = st.sidebar.selectbox("Bar labels", label_options)
    plot_kwargs["bar_label_mode"] = label_map[label_choice]
    if plot_kwargs["bar_label_mode"] == "value":
        decimals = st.sidebar.number_input("Decimal places", 0, 6, 2, 1, key="bar_label_decimals")
        plot_kwargs["bar_label_fmt"] = f"%.{decimals}f"
    plot_kwargs["xtick_rotation"] = st.sidebar.slider("Rotate X labels (°)", 0, 90, 0, 15)

    manual_ylim = st.sidebar.checkbox("Set Y-axis limits manually", value=False)
    if manual_ylim and params.get("y"):
        y_series = df[params["y"]]
        default_min = float(y_series.min())
        default_max = float(y_series.max())
        yc1, yc2 = st.sidebar.columns(2)
        plot_kwargs["y_min"] = yc1.number_input("Y min", value=default_min, format="%.3f")
        plot_kwargs["y_max"] = yc2.number_input("Y max", value=default_max, format="%.3f")
    else:
        plot_kwargs["y_min"] = None
        plot_kwargs["y_max"] = None
elif plot_name == "Box":
    plot_kwargs["show_points"] = st.sidebar.checkbox("Overlay data points", value=True)
elif plot_name == "Violin":
    plot_kwargs["inner"] = st.sidebar.selectbox("Inner annotation", ["box", "quartile", "point", "stick", None])
elif plot_name == "Histogram":
    plot_kwargs["bins"] = st.sidebar.slider("Bins", 5, 100, 30)
    plot_kwargs["kde"] = st.sidebar.checkbox("Overlay KDE", value=True)
    plot_kwargs["stat"] = st.sidebar.selectbox("Statistic", ["count", "density", "probability"])
elif plot_name == "Density (KDE)":
    plot_kwargs["fill"] = st.sidebar.checkbox("Fill", value=True)
elif plot_name == "Correlation heatmap":
    plot_kwargs["method"] = st.sidebar.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
    plot_kwargs["annotate"] = st.sidebar.checkbox("Annotate values", value=True)
    if plot_kwargs["annotate"]:
        plot_kwargs["value_decimals"] = st.sidebar.number_input(
            "Decimal places", 0, 6, 2, 1, key="corr_heatmap_decimals"
        )
    plot_kwargs["cmap"] = st.sidebar.selectbox("Colormap", HEATMAP_CMAPS)
elif plot_name == "Pairplot":
    plot_kwargs["diag_kind"] = st.sidebar.selectbox("Diagonal", ["kde", "hist"])
    plot_kwargs["height"] = st.sidebar.slider("Panel size (in)", 1.0, 3.0, 1.6, 0.1)
elif plot_name == "Settings heatmap":
    norm_labels = {"none": "Raw values", "minmax": "Min-max (0-1) per column", "zscore": "Z-score per column"}
    plot_kwargs["normalize"] = st.sidebar.selectbox(
        "Color scale", list(norm_labels.keys()), index=1, format_func=lambda v: norm_labels[v]
    )
    metric_cols_selected = params.get("metric_columns") or []
    invert_default = [c for c in metric_cols_selected if c.lower() in ("rmse", "mse", "mae", "loss", "error")]
    plot_kwargs["invert_columns"] = st.sidebar.multiselect(
        "Invert color (lower = better)", metric_cols_selected, default=invert_default
    )
    plot_kwargs["annotate"] = st.sidebar.checkbox("Annotate values", value=True)
    if plot_kwargs["annotate"]:
        decimals = st.sidebar.number_input("Decimal places", 0, 6, 3, 1, key="settings_heatmap_decimals")
        plot_kwargs["value_fmt"] = f".{decimals}f"
    plot_kwargs["cmap"] = st.sidebar.selectbox("Colormap", HEATMAP_CMAPS, key="settings_heatmap_cmap")

# ------------------------------------------------------------------- style
st.sidebar.title("5. Style (Nature-like)")
palette_name = st.sidebar.selectbox("Color palette", list(PALETTES.keys()))
font_scale = st.sidebar.slider("Font scale", 0.6, 2.0, 1.3, 0.1)
show_grid = st.sidebar.checkbox("Show gridlines", value=False)

width_preset = st.sidebar.selectbox("Figure width", list(FIGURE_WIDTHS_MM.keys()))
if FIGURE_WIDTHS_MM[width_preset] is not None:
    width_in = mm_to_inch(FIGURE_WIDTHS_MM[width_preset])
else:
    width_in = st.sidebar.slider("Custom width (in)", 2.0, 10.0, 4.0, 0.1)
height_in = st.sidebar.slider("Figure height (in)", 1.5, 8.0, 3.2, 0.1)

st.sidebar.title("6. Labels")
title_text = st.sidebar.text_input("Title", "")
xlabel_text = st.sidebar.text_input("X label override", "")
ylabel_text = st.sidebar.text_input("Y label override", "")

# ------------------------------------------------------------------ render
st.title("Plot Studio")

with st.expander("Data preview", expanded=False):
    st.write(f"**{df.shape[0]} rows × {df.shape[1]} columns**")
    st.dataframe(df.head(50), use_container_width=True)
    st.caption(f"dtypes: {dict(df.dtypes.astype(str))}")

apply_nature_style(font_scale=font_scale, show_grid=show_grid)
palette = PALETTES[palette_name]

func = spec["func"]
common = dict(palette=palette, title=title_text, xlabel=xlabel_text, ylabel=ylabel_text)
if plot_name == "Pairplot":
    common = dict(palette=palette, title=title_text)
    common["height"] = plot_kwargs.pop("height")
elif plot_name in ("Correlation heatmap", "Settings heatmap"):
    common = dict(title=title_text, xlabel=xlabel_text, ylabel=ylabel_text)
elif plot_name == "Line" and "color_map" in plot_kwargs:
    common["palette"] = plot_kwargs.pop("color_map")

figsize = (width_in, height_in)

call_kwargs = dict(**params, **plot_kwargs, **common)
if plot_name != "Pairplot":  # pairplot sizes panels via `height`, not `figsize`
    call_kwargs["figsize"] = figsize

try:
    fig = func(df, **call_kwargs)
except PlotError as e:
    st.error(str(e))
    st.stop()
except Exception as e:
    st.error(f"Could not build plot: {e}")
    st.stop()

st.pyplot(fig, use_container_width=False)

# ------------------------------------------------------------------ export
st.subheader("Export")
dpi = st.slider("Export DPI (raster formats)", 150, 1200, 300, 50)
c1, c2, c3 = st.columns(3)

for col, fmt, mime in [
    (c1, "png", "image/png"),
    (c2, "pdf", "application/pdf"),
    (c3, "svg", "image/svg+xml"),
]:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight")
    col.download_button(
        f"Download .{fmt}",
        data=buf.getvalue(),
        file_name=f"{plot_name.lower().replace(' ', '_')}.{fmt}",
        mime=mime,
        use_container_width=True,
    )

st.subheader("Python code")
st.caption("A standalone script that reproduces this exact figure — edit it freely, no app dependency.")
script = generate_script(plot_name, func, call_kwargs, font_scale, show_grid)
with st.expander("Show code", expanded=False):
    st.code(script, language="python")
st.download_button(
    "Download .py",
    data=script,
    file_name=f"{plot_name.lower().replace(' ', '_')}.py",
    mime="text/x-python",
)

plt.close(fig)

"""Nature-journal-inspired matplotlib/seaborn styling.

Loosely follows Nature's figure guidelines:
- sans-serif font (Arial/Helvetica fallback to DejaVu Sans)
- thin lines, small ticks, no top/right spine
- compact figure widths matching single/1.5/double column layouts
- colorblind-safe default palette (Okabe-Ito)
"""

import matplotlib.pyplot as plt
import seaborn as sns

# Okabe-Ito colorblind-safe palette, commonly used in Nature/Science figures.
OKABE_ITO = [
    "#000000",  # black
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
]

PALETTES = {
    "Okabe-Ito (colorblind-safe)": OKABE_ITO,
    "Muted": sns.color_palette("muted").as_hex(),
    "Deep": sns.color_palette("deep").as_hex(),
    "Viridis (sequential)": sns.color_palette("viridis", 8).as_hex(),
    "Rocket (sequential)": sns.color_palette("rocket", 8).as_hex(),
    "Grayscale": sns.color_palette("Greys", 8).as_hex()[2:],
}

# Column widths in mm converted to inches (Nature submission guidelines).
FIGURE_WIDTHS_MM = {
    "Single column (89 mm)": 89,
    "1.5 column (120 mm)": 120,
    "Double column (183 mm)": 183,
    "Custom": None,
}

FONT_STACK = ["Arial", "Helvetica", "DejaVu Sans", "Liberation Sans", "sans-serif"]

# Human-readable name -> matplotlib marker code.
MARKER_SHAPES = {
    "Circle": "o",
    "Square": "s",
    "Triangle up": "^",
    "Triangle down": "v",
    "Diamond": "D",
    "Thin diamond": "d",
    "Star": "*",
    "Pentagon": "p",
    "Hexagon": "h",
    "Plus": "P",
    "X": "X",
    "Cross (thin)": "x",
}


def mm_to_inch(mm: float) -> float:
    return mm / 25.4


def apply_nature_style(font_scale: float = 1.0, show_grid: bool = False) -> None:
    """Apply a Nature-like rcParams theme globally. Call before creating a figure."""
    base = 7 * font_scale  # Nature body text is ~7pt at final print size
    sns.set_theme(
        context="paper",
        style="ticks",
        font=FONT_STACK[0],
        rc={
            "font.sans-serif": FONT_STACK,
            "font.family": "sans-serif",
            "font.size": base,
            "axes.titlesize": base * 1.15,
            "axes.labelsize": base,
            "xtick.labelsize": base * 0.9,
            "ytick.labelsize": base * 0.9,
            "legend.fontsize": base * 0.9,
            "legend.title_fontsize": base,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.4,
            "ytick.minor.width": 0.4,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 1.2,
            "lines.markersize": 4,
            "patch.linewidth": 0.6,
            "axes.grid": show_grid,
            "grid.linewidth": 0.4,
            "grid.alpha": 0.4,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "figure.dpi": 150,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,  # editable text in Illustrator
            "ps.fonttype": 42,
        },
    )


def finalize_axes(ax, legend_outside: bool = True) -> None:
    """Common post-processing: trim spines, tidy legend placement."""
    sns.despine(ax=ax)
    legend = ax.get_legend()
    if legend is not None and legend_outside:
        legend.set_bbox_to_anchor((1.02, 1))
        legend.set_loc("upper left")
        legend.get_frame().set_linewidth(0.4)

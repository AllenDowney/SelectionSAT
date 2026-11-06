"""Plotting helpers and the default plot style for the SelectionSAT notebooks.

Importing this module applies the plot style (configure_plot_style), which the
figures in figs/ from selection_bias.ipynb were made with.
"""

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

# =============================================================================
# Matplotlib Configuration
# =============================================================================

AIBM_COLORS = {
    # AIBM brand colors
    "light_gray": "#F3F4F3",
    "medium_gray": "#767676",
    "green": "#0B8569",
    "light_green": "#AAC9B8",
    "orange": "#C55300",
    "light_orange": "#F4A26B",
    "purple": "#9657A5",
    "light_purple": "#CFBCD0",
    "blue": "#4575D6",
    "light_blue": "#C9D3E8",
    # Additional colors from coolers.co
    "dark_gray": "#404040",
    "dark_purple": "#28112B",
    "dark_green": "#002500",
    "amber": "#F5BB00",
    "oxford_blue": "#000022",
    "bittersweet": "#FF6666",
    "crimson": "#D62839",
}


def configure_plot_style():
    """Configure the default matplotlib style for AIBM plots.

    This function sets up the default style for plots, including:
    - Figure size and DPI
    - Color scheme
    - Font settings
    - Grid and spine settings
    - Tick settings

    The settings can be overridden for individual plots as needed.
    """
    # Figure size and DPI
    plt.rcParams["figure.dpi"] = 100
    plt.rcParams["figure.figsize"] = [8.75, 3.5]  # inches

    # Default color cycle
    colors = [
        AIBM_COLORS["green"],
        AIBM_COLORS["orange"],
        AIBM_COLORS["blue"],
        AIBM_COLORS["purple"],
    ]
    cycler = plt.cycler(color=colors)
    plt.rc("axes", prop_cycle=cycler)

    # Font settings
    # Try to use PT Sans, fall back to system fonts if not available
    if "PT Sans" in [f.name for f in font_manager.fontManager.ttflist]:
        plt.rcParams["font.family"] = "PT Sans"
    else:
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = [
            "DejaVu Sans",
            "Arial",
            "Helvetica",
            "sans-serif",
        ]

    plt.rcParams["legend.fontsize"] = "small"

    # Tick and label colors
    plt.rcParams["axes.edgecolor"] = AIBM_COLORS["medium_gray"]
    plt.rcParams["xtick.color"] = AIBM_COLORS["medium_gray"]
    plt.rcParams["ytick.color"] = AIBM_COLORS["medium_gray"]
    plt.rcParams["axes.labelcolor"] = AIBM_COLORS["medium_gray"]

    # Default spine settings (can be overridden per plot)
    plt.rcParams["axes.spines.top"] = False
    plt.rcParams["axes.spines.right"] = False
    plt.rcParams["axes.spines.left"] = False
    plt.rcParams["axes.spines.bottom"] = False

    # Default grid settings (can be overridden per plot)
    plt.rcParams["grid.color"] = AIBM_COLORS["light_gray"]
    plt.rcParams["grid.linestyle"] = "-"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["axes.grid"] = False  # Disable grid by default
    plt.rcParams["axes.grid.axis"] = "y"

    # Tick mark settings
    plt.rcParams["xtick.major.size"] = 0
    plt.rcParams["xtick.minor.size"] = 0
    plt.rcParams["ytick.major.size"] = 0
    plt.rcParams["ytick.minor.size"] = 0


# Apply the default style
configure_plot_style()


def underride(d, **options):
    """Add key-value pairs to d only if key is not in d.

    Args:
        d: dictionary
        options: keyword args to add to d

    Returns:
        Updated dictionary
    """
    for key, val in options.items():
        d.setdefault(key, val)
    return d


def value_counts(seq, **options):
    """Make a series of values and the number of times they appear.

    Returns a DataFrame because they get rendered better in Jupyter.

    Args:
        seq: sequence
        options: passed to pd.Series.value_counts

    Returns:
        pd.DataFrame with value counts
    """
    options = underride(options, dropna=False)
    series = pd.Series(seq).value_counts(**options).sort_index()
    series.index.name = "values"
    series.name = "counts"
    return pd.DataFrame(series)


def decorate(**options):
    """Decorate the current axes.

    Call decorate with keyword arguments like
    decorate(title='Title',
             xlabel='x',
             ylabel='y')

    The keyword arguments can be any of the axis properties
    https://matplotlib.org/api/axes_api.html
    """
    legend = options.pop("legend", True)
    loc = options.pop("loc", "best")
    ax = plt.gca()
    ax.set(**options)

    handles, labels = ax.get_legend_handles_labels()
    if handles and legend:
        ax.legend(handles, labels, loc=loc)

    plt.tight_layout()

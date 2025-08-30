"""Utility functions for data analysis and visualization."""

import os
import re
from matplotlib import font_manager

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.stats import beta, norm

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
        AIBM_COLORS["orange"],
        AIBM_COLORS["green"],
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
    plt.rcParams["axes.spines.left"] = True  # Keep left spine by default
    plt.rcParams["axes.spines.bottom"] = True  # Keep bottom spine by default

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

# =============================================================================
# File I/O Functions
# =============================================================================


def write_table(table, label, **options):
    """Write a table in LaTex format.

    Args:
        table: DataFrame
        label: string
        options: passed to DataFrame.to_latex
    """
    filename = f"tables/{label}.tex"
    os.makedirs("tables", exist_ok=True)
    with open(filename, "w", encoding="utf8") as fp:
        s = table.to_latex(**options)
        fp.write(s)


def write_pmf(pmf, label):
    """Write a Pmf object as a table.

    Args:
        pmf: Pmf
        label: string
    """
    df = pd.DataFrame()
    df["qs"] = pmf.index
    df["ps"] = pmf.values
    write_table(df, label, index=False)


def savefig(prefix, fig_number, extra_artists=[], dpi=150):
    """Save the current figure with the given filename.

    Args:
        prefix: string prefix for filename
        fig_number: The figure number
        extra_artists: List of additional artist to include in the bounding box
        dpi: Dots per inch for the saved image
    """
    filename = f"{prefix}{fig_number:02d}"
    if extra_artists:
        plt.savefig(
            filename, dpi=dpi, bbox_inches="tight", bbox_extra_artists=extra_artists
        )
    else:
        plt.savefig(filename, dpi=dpi)


# =============================================================================
# Data Manipulation Functions
# =============================================================================


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


def value_count_frame(data, columns, normalize=False):
    """Make a DataFrame of value counts.

    Args:
        data: DataFrame
        columns: list of column names
        normalize: whether to normalize the counts

    Returns:
        DataFrame with value counts
    """
    dfs = []
    for col in columns:
        df = value_counts(data[col], normalize=normalize)
        df.columns = [col]
        dfs.append(df)
    return pd.concat(dfs, axis=1)


def find_columns(df, prefix):
    """Find columns that start with a given prefix.

    Args:
        df: DataFrame
        prefix: string prefix

    Returns:
        list of column names
    """
    return [
        col
        for col in df.columns
        if col.startswith(prefix)
        and not col.endswith("skp")
        and not col.endswith("timing")
    ]


def round_into_bins(series, bin_width, low=0, high=None):
    """Rounds values down to the bin they belong in.

    series: pd.Series
    bin_width: number, width of the bins

    returns: Series of bin values (with NaN preserved)
    """
    if high is None:
        high = series.max()

    bins = np.arange(low, high + bin_width, bin_width)
    indices = np.digitize(series, bins)
    result = pd.Series(bins[indices - 1], index=series.index, dtype="float")

    result[series.isna()] = np.nan
    return result


# =============================================================================
# Categorical Data Functions
# =============================================================================


def extract_categorical_mapping(series):
    """Extract a mapping from categorical codes to descriptions.

    Args:
        series: pandas Series

    Returns:
        pd.Series mapping from codes to descriptions
    """
    mapping = {}

    for item in series.unique():  # Process unique categorical values
        match = re.match(r"([-\d]+)\.\s(.+)", str(item).strip())
        if match:
            code, description = match.groups()
            mapping[int(code)] = description

    return pd.Series(mapping).sort_index()


def make_categorical_mappings(df, skip_cols=["age"]):
    """Make a mapping from variable names to dictionaries of codes and values.

    Args:
        df: DataFrame
        skip_cols: list of string column names to skip

    Returns:
        dictionary that maps from column names to dictionaries
    """
    mappings = {}
    for col in df.columns:
        if col in skip_cols:
            continue
        mapping = extract_categorical_mapping(df[col])
        if len(mapping) > 0:
            mappings[col] = mapping
    return mappings


def map_codes_to_categories(cat_series: pd.Series, code_series: pd.Series) -> pd.Series:
    """Map numeric codes to category labels.

    Args:
        cat_series: Series containing category labels
        code_series: Series containing numeric codes

    Returns:
        Series with mapped category labels
    """
    # Extract the mapping
    mapping = extract_categorical_mapping(cat_series)

    # Map the codes to categories
    return code_series.map(mapping)


# =============================================================================
# Statistical Functions
# =============================================================================


def estimate_proportion_jeffreys(success_series, confidence_level=0.95):
    """Estimate proportion using Jeffreys prior.

    Args:
        success_series: Boolean series (True = success)
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        tuple: (proportion, lower_bound, upper_bound)
    """
    success_series = success_series.astype(float)
    n = len(success_series)
    k = success_series.sum()

    # Jeffreys prior: Beta(0.5, 0.5)
    alpha = k + 0.5
    beta = n - k + 0.5

    # Calculate posterior mean
    proportion = alpha / (alpha + beta)

    # Calculate credible interval
    lower = beta.ppf((1 - confidence_level) / 2, alpha, beta)
    upper = beta.ppf(1 - (1 - confidence_level) / 2, alpha, beta)

    return proportion, lower, upper


def estimate_proportion_wilson(success_series, weights_series, confidence_level=0.95):
    """Estimate weighted proportion with Wilson score interval adjusted using effective sample size.

    Args:
        success_series: Boolean series (True = success)
        weights_series: Corresponding weights
        confidence_level: Confidence level (e.g., 0.95)

    Returns:
        tuple: (weighted_proportion, lower_bound, upper_bound)
    """
    success_series = success_series.astype(float)
    weights_series = weights_series.astype(float)

    weighted_successes = (success_series * weights_series).sum()
    total_weight = weights_series.sum()

    # Estimate effective sample size
    n_eff = total_weight**2 / (weights_series**2).sum()

    # Z-score for confidence interval
    z = norm.ppf(1 - (1 - confidence_level) / 2)

    denominator = 1 + z**2 / n_eff
    center = (p + z**2 / (2 * n_eff)) / denominator
    margin = (z * np.sqrt((p * (1 - p) + z**2 / (4 * n_eff)) / n_eff)) / denominator

    lower = center - margin
    upper = center + margin

    return p, lower, upper


def estimate_columns(df, columns, values):
    """Estimate proportions for multiple columns.

    Args:
        df: DataFrame
        columns: list of column names
        values: list of values to estimate

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for col in columns:
        for value in values:
            success = df[col] == value
            p, lower, upper = estimate_proportion_wilson(success, df["weight"])
            estimates.append(
                {
                    "column": col,
                    "value": value,
                    "proportion": p,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pd.DataFrame(estimates)


def estimate_value_map(df, columns, value_map):
    """Estimate proportions using a value mapping.

    Args:
        df: DataFrame
        columns: list of column names
        value_map: dictionary mapping values to labels

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for col in columns:
        for value, label in value_map.items():
            success = df[col] == value
            p, lower, upper = estimate_proportion_wilson(success, df["weight"])
            estimates.append(
                {
                    "column": col,
                    "value": value,
                    "label": label,
                    "proportion": p,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pd.DataFrame(estimates)


def estimate_gender_map(columns, gender_map, value_map):
    """Estimate proportions by gender.

    Args:
        columns: list of column names
        gender_map: dictionary mapping gender codes to labels
        value_map: dictionary mapping values to labels

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for col in columns:
        for gender, gender_label in gender_map.items():
            for value, label in value_map.items():
                success = (df[col] == value) & (df["gender"] == gender)
                p, lower, upper = estimate_proportion_wilson(success, df["weight"])
                estimates.append(
                    {
                        "column": col,
                        "gender": gender,
                        "gender_label": gender_label,
                        "value": value,
                        "label": label,
                        "proportion": p,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    return pd.DataFrame(estimates)


def estimate_ordinal(df, column, values, cumulative=False, confidence_level=0.84):
    """Estimate proportions for ordinal data.

    Args:
        df: DataFrame
        column: column name
        values: list of values
        cumulative: whether to compute cumulative proportions
        confidence_level: confidence level

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for value in values:
        if cumulative:
            success = df[column] >= value
        else:
            success = df[column] == value
        p, lower, upper = estimate_proportion_wilson(success, df["weight"])
        estimates.append(
            {
                "value": value,
                "proportion": p,
                "lower": lower,
                "upper": upper,
            }
        )
    return pd.DataFrame(estimates)


def ordinal_gender_map(
    gender_map, column, values, cumulative=False, confidence_level=0.84
):
    """Estimate ordinal proportions by gender.

    Args:
        gender_map: dictionary mapping gender codes to labels
        column: column name
        values: list of values
        cumulative: whether to compute cumulative proportions
        confidence_level: confidence level

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for gender, gender_label in gender_map.items():
        for value in values:
            if cumulative:
                success = (df[column] >= value) & (df["gender"] == gender)
            else:
                success = (df[column] == value) & (df["gender"] == gender)
            p, lower, upper = estimate_proportion_wilson(success, df["weight"])
            estimates.append(
                {
                    "gender": gender,
                    "gender_label": gender_label,
                    "value": value,
                    "proportion": p,
                    "lower": lower,
                    "upper": upper,
                }
            )
    return pd.DataFrame(estimates)


def ordinal_age_gender_map(
    age_map, column, values, cumulative=False, confidence_level=0.84
):
    """Estimate ordinal proportions by age and gender.

    Args:
        age_map: dictionary mapping age codes to labels
        column: column name
        values: list of values
        cumulative: whether to compute cumulative proportions
        confidence_level: confidence level

    Returns:
        DataFrame with estimates
    """
    estimates = []
    for age, age_label in age_map.items():
        for gender, gender_label in gender_map.items():
            for value in values:
                if cumulative:
                    success = (
                        (df[column] >= value)
                        & (df["age"] == age)
                        & (df["gender"] == gender)
                    )
                else:
                    success = (
                        (df[column] == value)
                        & (df["age"] == age)
                        & (df["gender"] == gender)
                    )
                p, lower, upper = estimate_proportion_wilson(success, df["weight"])
                estimates.append(
                    {
                        "age": age,
                        "age_label": age_label,
                        "gender": gender,
                        "gender_label": gender_label,
                        "value": value,
                        "proportion": p,
                        "lower": lower,
                        "upper": upper,
                    }
                )
    return pd.DataFrame(estimates)


# =============================================================================
# Basic Plotting Functions
# =============================================================================


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


def anchor_legend(x, y):
    """Place the upper left corner of the legend box.

    Args:
        x: x coordinate
        y: y coordinate
    """
    plt.legend(bbox_to_anchor=(x, y), loc="upper left", ncol=1)
    plt.tight_layout()


def add_text(x, y, text, **options):
    """Add text to the current axes.

    Args:
        x: float
        y: float
        text: string
        options: keyword arguments passed to plt.text
    """
    ax = plt.gca()
    underride(
        options,
        transform=ax.transAxes,
        color="0.2",
        ha="left",
        va="bottom",
        fontsize=9,
    )
    plt.text(x, y, text, **options)


def remove_spines():
    """Remove the spines of a plot but keep the ticks visible."""
    ax = plt.gca()
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Ensure ticks stay visible
    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("left")


def add_logo(filename="logo-hq-small.png", location=(1.0, -0.35), size=(0.5, 0.25)):
    """Add a logo inside an inset axis positioned relative to the main plot.

    Args:
        filename: path to logo image
        location: tuple of (x, y) coordinates
        size: tuple of (width, height)

    Returns:
        The inset axis containing the logo
    """
    logo = mpimg.imread(filename)

    # Create an inset axis in the given location
    ax = plt.gca()
    fig = ax.figure
    ax_inset = inset_axes(
        ax,
        width=size[0],
        height=size[1],
        loc="lower right",
        bbox_to_anchor=location,
        bbox_transform=fig.transFigure,
        borderpad=0,
    )

    # Display the logo
    ax_inset.imshow(logo)
    ax_inset.axis("off")
    
    # Restore the original axes as current
    plt.sca(ax)

    return ax_inset


def add_subtext(text, x=0, y=-0.35):
    """Add a text label below the current plot.

    Args:
        text: string
        x: x coordinate
        y: y coordinate

    Returns:
        The text object
    """
    ax = plt.gca()
    fig = ax.figure
    return plt.figtext(
        x, y, text, ha="left", va="bottom", fontsize=8, transform=fig.transFigure
    )


def add_title(title, subtitle, pad=25, x=0, y=1.02):
    """Add a title and subtitle to the current plot.

    Args:
        title: Title of the plot
        subtitle: Subtitle of the plot
        pad: Padding between the title and subtitle
        x: x coordinate for subtitle
        y: y coordinate for subtitle
    """
    plt.title(title, loc="left", pad=pad)
    add_text(x, y, subtitle)


def reverse_color_map(color_map):
    """Reverse the order of colors in a color map.

    Args:
        color_map: dictionary mapping values to colors

    Returns:
        dictionary with reversed color order
    """
    return {k: v for k, v in reversed(list(color_map.items()))}


# =============================================================================
# Survey Data Visualization Functions
# =============================================================================


def plot_responses(
    summary, gender, response, issue_names, style, label_response=True, **options
):
    """Plot survey responses.

    Args:
        summary: DataFrame with response data
        gender: gender code
        response: response code
        issue_names: list of issue names
        style: dictionary of style parameters
        label_response: whether to label the response
        options: additional plotting options
    """
    # Filter data for this gender and response
    data = summary[
        (summary["gender"] == gender) & (summary["value"] == response)
    ].copy()

    # Plot each issue
    for i, issue in enumerate(issue_names):
        row = data[data["column"] == issue].iloc[0]
        plot_estimate(i, row, style, label_response, **options)


def plot_responses_by_gender(summary, response, issue_names, **options):
    """Plot survey responses by gender.

    Args:
        summary: DataFrame with response data
        response: response code
        issue_names: list of issue names
        options: additional plotting options
    """
    # Define styles for each gender
    styles = {
        1: dict(color=AIBM_COLORS["blue"], label="Men"),
        2: dict(color=AIBM_COLORS["orange"], label="Women"),
    }

    # Plot responses for each gender
    for gender in [1, 2]:
        plot_responses(
            summary, gender, response, issue_names, styles[gender], **options
        )


def stacked_bar_chart(y, estimate, color_map, **options):
    """Create a stacked bar chart.

    Args:
        y: y coordinate
        estimate: DataFrame with estimates
        color_map: dictionary mapping values to colors
        options: additional plotting options
    """
    # Plot each segment
    for value, color in color_map.items():
        row = estimate[estimate["value"] == value].iloc[0]
        plt.barh(y, row["proportion"], color=color, **options)


def plot_age_gender_summary(
    summary, age_map, group_name_map, color_map, response_map, y=0
):
    """Plot summary by age and gender.

    Args:
        summary: DataFrame with summary data
        age_map: dictionary mapping age codes to labels
        group_name_map: dictionary mapping group codes to names
        color_map: dictionary mapping values to colors
        response_map: dictionary mapping response codes to labels
        y: starting y coordinate
    """
    # Plot each age group
    for age, age_label in age_map.items():
        # Plot each gender
        for gender, gender_label in group_name_map.items():
            # Filter data for this age and gender
            data = summary[
                (summary["age"] == age) & (summary["gender"] == gender)
            ].copy()

            # Plot responses
            for response, label in response_map.items():
                row = data[data["value"] == response].iloc[0]
                plot_estimate(y, row, color_map[response], label, **options)
                y += 1


def plot_estimate(y, row, style, label, **options):
    """Plot a single estimate.

    Args:
        y: y coordinate
        row: DataFrame row with estimate data
        style: dictionary of style parameters
        label: label for the estimate
        options: additional plotting options
    """
    # Plot the estimate
    plt.barh(y, row["proportion"], **style, **options)

    # Add error bars
    plt.errorbar(
        row["proportion"],
        y,
        xerr=[[row["proportion"] - row["lower"]], [row["upper"] - row["proportion"]]],
        fmt="none",
        color="black",
        capsize=3,
    )

    # Add label
    if label:
        plt.text(
            row["proportion"] + 0.01,
            y,
            label,
            va="center",
            ha="left",
            fontsize=9,
        )


def plot_estimates(estimate, style, label, **options):
    """Plot multiple estimates.

    Args:
        estimate: DataFrame with estimates
        style: dictionary of style parameters
        label: label for the estimates
        options: additional plotting options
    """
    # Plot each estimate
    for i, row in estimate.iterrows():
        plot_estimate(i, row, style, label, **options)


def add_responses(response_map):
    """Add response labels to the plot.

    Args:
        response_map: dictionary mapping response codes to labels
    """
    # Add each response label
    for response, label in response_map.items():
        plt.text(
            0,
            response,
            label,
            va="center",
            ha="right",
            fontsize=9,
        )


def plot_estimates_by_age_gender(summary, age_map, group_name_map, **options):
    """Plot estimates by age and gender.

    Args:
        summary: DataFrame with summary data
        age_map: dictionary mapping age codes to labels
        group_name_map: dictionary mapping group codes to names
        options: additional plotting options
    """
    # Plot each age group
    for age, age_label in age_map.items():
        # Plot each gender
        for gender, gender_label in group_name_map.items():
            # Filter data for this age and gender
            data = summary[
                (summary["age"] == age) & (summary["gender"] == gender)
            ].copy()

            # Plot estimates
            plot_estimates(data, gender_label, **options)


code_to_wef_country = {
    "ALB": "Albania",
    "DZA": "Algeria",
    "AGO": "Angola",
    "ARG": "Argentina",
    "ARM": "Armenia",
    "AUS": "Australia",
    "AUT": "Austria",
    "AZE": "Azerbaijan",
    "BHR": "Bahrain",
    "BGD": "Bangladesh",
    "BRB": "Barbados",
    "BLR": "Belarus",
    "BEL": "Belgium",
    "BLZ": "Belize",
    "BEN": "Benin",
    "BTN": "Bhutan",
    "BOL": "Bolivia",
    "BIH": "Bosnia-Herzegovina",
    "BWA": "Botswana",
    "BRA": "Brazil",
    "BRN": "Brunei",
    "BGR": "Bulgaria",
    "BFA": "Burkina Faso",
    "BDI": "Burundi",
    "KHM": "Cambodia",
    "CMR": "Cameroon",
    "CAN": "Canada",
    "CPV": "Cape Verde",
    "TCD": "Chad",
    "CHL": "Chile",
    "CHN": "China",
    "COL": "Colombia",
    "COM": "Comoros",
    "COD": "D.R. Congo",
    "CRI": "Costa Rica",
    "CIV": "Côte D'Ivoire",
    "HRV": "Croatia",
    "CYP": "Cyprus",
    "CZE": "Czechia",
    "DNK": "Denmark",
    "DOM": "Dominican Republic",
    "ECU": "Ecuador",
    "EGY": "Egypt",
    "SLV": "El Salvador",
    "EST": "Estonia",
    "SWZ": "Eswatini",
    "ETH": "Ethiopia",
    "FJI": "Fiji",
    "FIN": "Finland",
    "FRA": "France",
    "GMB": "Gambia",
    "GEO": "Georgia",
    "DEU": "Germany",
    "GHA": "Ghana",
    "GRC": "Greece",
    "GTM": "Guatemala",
    "GIN": "Guinea",
    "GUY": "Guyana",
    "HND": "Honduras",
    "HUN": "Hungary",
    "ISL": "Iceland",
    "IND": "India",
    "IDN": "Indonesia",
    "IRN": "Iran",
    "IRL": "Ireland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JAM": "Jamaica",
    "JPN": "Japan",
    "JOR": "Jordan",
    "KAZ": "Kazakhstan",
    "KEN": "Kenya",
    "KWT": "Kuwait",
    "KGZ": "Kyrgyzstan",
    "LAO": "Laos",
    "LVA": "Latvia",
    "LBN": "Lebanon",
    "LSO": "Lesotho",
    "LBR": "Liberia",
    "LTU": "Lithuania",
    "LUX": "Luxembourg",
    "MDG": "Madagascar",
    "MYS": "Malaysia",
    "MDV": "Maldives",
    "MLI": "Mali",
    "MLT": "Malta",
    "MUS": "Mauritius",
    "MEX": "Mexico",
    "MDA": "Moldova",
    "MNG": "Mongolia",
    "MNE": "Montenegro",
    "MAR": "Morocco",
    "MOZ": "Mozambique",
    "NAM": "Namibia",
    "NPL": "Nepal",
    "NLD": "Netherlands",
    "NZL": "New Zealand",
    "NIC": "Nicaragua",
    "NER": "Niger",
    "NGA": "Nigeria",
    "MKD": "North Macedonia",
    "NOR": "Norway",
    "OMN": "Oman",
    "PAK": "Pakistan",
    "PAN": "Panama",
    "PRY": "Paraguay",
    "PER": "Peru",
    "PHL": "Philippines",
    "POL": "Poland",
    "PRT": "Portugal",
    "QAT": "Qatar",
    "ROU": "Romania",
    "RWA": "Rwanda",
    "SAU": "Saudi Arabia",
    "SEN": "Senegal",
    "SRB": "Serbia",
    "SLE": "Sierra Leone",
    "SGP": "Singapore",
    "SVK": "Slovakia",
    "SVN": "Slovenia",
    "ZAF": "South Africa",
    "KOR": "South Korea",
    "ESP": "Spain",
    "LKA": "Sri Lanka",
    "SDN": "Sudan",
    "SUR": "Suriname",
    "SWE": "Sweden",
    "CHE": "Switzerland",
    "TJK": "Tajikistan",
    "THA": "Thailand",
    "TLS": "Timor-Leste",
    "TGO": "Togo",
    "TUN": "Tunisia",
    "TUR": "Türkiye",
    "UGA": "Uganda",
    "UKR": "Ukraine",
    "ARE": "United Arab Emirates",
    "GBR": "United Kingdom",
    "TZA": "Tanzania",
    "USA": "United States",
    "URY": "Uruguay",
    "UZB": "Uzbekistan",
    "VUT": "Vanuatu",
    "VNM": "Vietnam",
    "ZMB": "Zambia",
    "ZWE": "Zimbabwe",
    # Additional countries not in original list
    "ABW": "Aruba",
    "CAF": "Central African Republic",
    "COG": "Congo",
    "CUB": "Cuba",
    "CYM": "Cayman Islands",
    "ERI": "Eritrea",
    "GAB": "Gabon",
    "GNB": "Guinea-Bissau",
    "GNQ": "Equatorial Guinea",
    "GUM": "Guam",
    "HTI": "Haiti",
    "MAC": "Macao",
    "MMR": "Myanmar",
    "MRT": "Mauritania",
    "MWI": "Malawi",
    "NCL": "New Caledonia",
    "PNG": "Papua New Guinea",
    "PRI": "Puerto Rico",
    "PRK": "North Korea",
    "SSD": "South Sudan",
    "STP": "São Tomé and Príncipe",
    "SYC": "Seychelles",
    "SYR": "Syria",
    "TKM": "Turkmenistan",
    "TTO": "Trinidad and Tobago",
    "VEN": "Venezuela",
    "YEM": "Yemen",
}

wef_country_to_code = {country: code for code, country in code_to_wef_country.items()}

oecd_codes = ['AUS', 'AUT', 'BEL', 'CAN', 'CHE', 'CHL', 'COL', 'CRI', 'CZE',
              'DEU', 'DNK', 'ESP', 'EST', 'FIN', 'FRA', 'GBR', 'GRC', 'HUN', 'IRL',
              'ISL', 'ISR', 'ITA', 'JPN', 'KOR', 'LTU', 'LUX', 'LVA', 'MEX',
              'NLD', 'NOR', 'NZL', 'POL', 'PRT', 'SVK', 'SVN', 'SWE', 'TUR', 'USA']




def read_wef_file(filename):
    """Read the WEF file and return a DataFrame.
    
    Args:
        filename: name of the WEF file
    """
    df = pd.read_csv(filename)

    df["country"] = df["country"].replace(
        {
            "United States of America": "United States",
            "Brunei Darussalam": "Brunei",
            "Moldova, Republic of": "Moldova",
            "Congo, Democratic Republic of t": "D.R. Congo",
            "United Republic of Tanzania": "Tanzania",
            "Viet Nam": "Vietnam",
            "Bosnia and Herzegovina": "Bosnia-Herzegovina",
            "Lao PDR": "Laos",
        }
    )
    df.index = df["country"].map(wef_country_to_code)
    df.index.name = "code"
    return df


def plot_revised_scores(df):
    """Plot revised scores for countries.
        
    Args:
        df: DataFrame with revised scores
    """
    n = len(df)
    height = 15 * n / 100
    fig, ax = plt.subplots(figsize=(6, height))
    plt.hlines(
        df["country"], df["score"], df["revised_score"], color=AIBM_COLORS["light_gray"]
    )
    plt.plot(df["score"], df["country"], "|", color=AIBM_COLORS["blue"])
    plt.plot(df["revised_score"], df["country"], "<", color=AIBM_COLORS["blue"])
    ax.invert_yaxis()
    plt.ylim(n + 1, -1)
    embolden_countries(['United States'])

def plot_revised_ranks(df):
    """Plot revised ranks for countries.
        
    Args:
        df: DataFrame with revised ranks
    """
    n = len(df)
    height = 15 * n / 100
    fig, ax = plt.subplots(figsize=(6, height))
    plt.hlines(
        df["country"], df["rank"], df["revised_rank"], color=AIBM_COLORS["light_gray"]
    )
    plt.plot(df["rank"], df["country"], "|", color=AIBM_COLORS["blue"])
    plt.plot(df["revised_rank"], df["country"], "o", color=AIBM_COLORS["blue"])
    ax.invert_yaxis()
    plt.ylim(n + 1, -1)
    embolden_countries(['United States'])
    
def embolden_countries(countries):
    # Make selected countries bold
    ax = plt.gca()
    ytick_labels = ax.get_yticklabels()
    for i, label in enumerate(ytick_labels):
        if label.get_text() in countries:
            plt.setp(label, fontweight='bold')


def plot_score_distributions(df, **options):
    kde_options = dict(cut=0, bw_adjust=0.7)

    sns.kdeplot(df['score'], label='WEF truncated scores', **kde_options)
    sns.kdeplot(df['revised_score'], label='Revised symmetric scores', **kde_options)

    decorate(**options)
    add_subtext("Source: World Economic Forum", y=-0.25)
    logo = add_logo(location=(1.0, -0.25))


def make_weights(column, label):
    std = column.std() 
    weights = pd.DataFrame(std, index=[label], columns=['std'])
    weights['inv std'] = 0.01 / std
    return weights

def make_weight_table(table, label):
    weights_orig = make_weights(table['score'], label)
    weights_revised = make_weights(table['revised_score'], label)
    weights = pd.concat([weights_orig, weights_revised],
                        axis=1, 
                        keys=['original', 'revised'])
    return weights

def make_rank_table(df):
    columns = ['country', 'ratio', 'rank', 'revised_rank', 'score', 'revised_score']
    df['revised_rank'] = df['revised_score'].rank(method='min', ascending=False)
    table = df[columns]
    return table

def plot_percentages(df):
    df['male'] = df['left'].where(df['score'] == 1, df['right'])
    df['female'] = df['right'].where(df['score'] == 1, df['left'])

    plot_indicators(df)


def plot_indicators(df):
    
    df_sorted = df.sort_values(by='female', ascending=False)
    country = df_sorted['country']
    male = df_sorted['male']
    female = df_sorted['female']

    fig, ax = plt.subplots(figsize=(6, 6))
    plt.hlines(country, male, female, color=AIBM_COLORS['light_gray'])
    plt.plot(male, country, 's', color=AIBM_COLORS['green'], label='Male')
    plt.plot(female, country, 'o', color=AIBM_COLORS['orange'], label='Female')
    
    n = len(df_sorted)
    decorate(ylim=[n + 0.5, -0.5])

    add_subtext("Source: WEF Global Gender Gap Report", y=-0.05)
    logo = add_logo(location=(1.0, -0.05))
    embolden_countries(['United States'])

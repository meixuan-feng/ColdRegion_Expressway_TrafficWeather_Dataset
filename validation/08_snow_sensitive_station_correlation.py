# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import pearsonr


COLUMN_CANDIDATES = {
    "station": ["station_id", "raw_station_code", "station_code"],
    "volume": ["volume_total_veh", "total_volume", "volume_total", "total_flow"],
    "speed": ["traffic_speed_kmh", "avg_speed", "average_speed", "speed_kmh", "speed"],
    "temperature": ["temperature_c", "temperature_2m", "t2m"],
    "snow_depth": ["snow_depth_mm", "snow_depth_cumulative_mm", "snow_depth"],
}


VARIABLE_LABELS = {
    "volume": "Volume\n(veh/h)",
    "speed": "Speed\n(km/h)",
    "temperature": "Temperature\n(°C)",
    "snow_depth": "Snow depth\n(mm)",
}


THOUSAND_VARIABLES = {
    "Volume\n(veh/h)",
    "Snow depth\n(mm)",
}


CORRELATION_CMAP = LinearSegmentedColormap.from_list(
    "soft_diverging",
    ["#AFCBE3", "#F7F7F7", "#F2C6B4"]
)


def set_publication_style(font_name="Arial"):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 7
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 6
    plt.rcParams["ytick.labelsize"] = 6
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"
    plt.rcParams["axes.unicode_minus"] = False


def resolve_column(columns, candidates, specified_col=None, target_name=None):
    if specified_col is not None:
        if specified_col not in columns:
            raise KeyError(f"Column not found for {target_name}: {specified_col}")
        return specified_col

    lower_map = {str(col).lower(): col for col in columns}

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    raise KeyError(
        f"No column was detected for {target_name}. "
        f"Available columns are: {list(columns)}"
    )


def format_p_value(p_value):
    if p_value < 0.001:
        return "p < 0.001"
    return f"p = {p_value:.6g}"


def prepare_snow_sensitive_data(
    input_file,
    station_col=None,
    volume_col=None,
    speed_col=None,
    temperature_col=None,
    snow_depth_col=None,
    snow_threshold_mm=20.0,
    speed_min=10.0,
    speed_max=140.0,
    min_snow_samples=5,
    min_clear_samples=5,
    top_k=5,
    max_plot_points=5000,
    random_state=42
):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0)

    resolved_cols = {
        "station": resolve_column(
            header.columns,
            COLUMN_CANDIDATES["station"],
            specified_col=station_col,
            target_name="station"
        ),
        "volume": resolve_column(
            header.columns,
            COLUMN_CANDIDATES["volume"],
            specified_col=volume_col,
            target_name="traffic volume"
        ),
        "speed": resolve_column(
            header.columns,
            COLUMN_CANDIDATES["speed"],
            specified_col=speed_col,
            target_name="traffic speed"
        ),
        "temperature": resolve_column(
            header.columns,
            COLUMN_CANDIDATES["temperature"],
            specified_col=temperature_col,
            target_name="temperature"
        ),
        "snow_depth": resolve_column(
            header.columns,
            COLUMN_CANDIDATES["snow_depth"],
            specified_col=snow_depth_col,
            target_name="snow depth"
        ),
    }

    data = pd.read_csv(
        input_path,
        usecols=list(resolved_cols.values()),
        low_memory=False
    )

    data = data.rename(
        columns={
            resolved_cols["station"]: "station_id",
            resolved_cols["volume"]: VARIABLE_LABELS["volume"],
            resolved_cols["speed"]: VARIABLE_LABELS["speed"],
            resolved_cols["temperature"]: VARIABLE_LABELS["temperature"],
            resolved_cols["snow_depth"]: VARIABLE_LABELS["snow_depth"],
        }
    )

    data["station_id"] = data["station_id"].astype(str).str.strip()

    for col in [
        VARIABLE_LABELS["volume"],
        VARIABLE_LABELS["speed"],
        VARIABLE_LABELS["temperature"],
        VARIABLE_LABELS["snow_depth"],
    ]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    if data[VARIABLE_LABELS["temperature"]].median(skipna=True) > 100:
        data[VARIABLE_LABELS["temperature"]] = data[VARIABLE_LABELS["temperature"]] - 273.15

    data = data.dropna(
        subset=[
            "station_id",
            VARIABLE_LABELS["volume"],
            VARIABLE_LABELS["speed"],
            VARIABLE_LABELS["temperature"],
            VARIABLE_LABELS["snow_depth"],
        ]
    ).copy()

    data = data[
        (data[VARIABLE_LABELS["speed"]] > speed_min) &
        (data[VARIABLE_LABELS["speed"]] < speed_max)
    ].copy()

    data = data[data[VARIABLE_LABELS["volume"]] >= 0].copy()
    data = data[data[VARIABLE_LABELS["snow_depth"]] >= 0].copy()

    data["Is snow"] = data[VARIABLE_LABELS["snow_depth"]] > snow_threshold_mm

    sample_counts = (
        data.groupby(["station_id", "Is snow"])
        .size()
        .unstack(fill_value=0)
    )

    if True not in sample_counts.columns:
        sample_counts[True] = 0

    if False not in sample_counts.columns:
        sample_counts[False] = 0

    valid_station_ids = sample_counts[
        (sample_counts[True] >= min_snow_samples) &
        (sample_counts[False] >= min_clear_samples)
    ].index.tolist()

    if len(valid_station_ids) == 0:
        raise RuntimeError(
            "No station satisfies both snow and non-snow sample-size requirements."
        )

    speed_stats = (
        data[data["station_id"].isin(valid_station_ids)]
        .groupby(["station_id", "Is snow"])[VARIABLE_LABELS["speed"]]
        .mean()
        .unstack()
    )

    if True not in speed_stats.columns or False not in speed_stats.columns:
        raise RuntimeError("Unable to obtain both snow and non-snow mean speeds.")

    speed_stats["relative_speed_loss"] = (
        speed_stats[False] - speed_stats[True]
    ) / speed_stats[False]

    speed_stats = (
        speed_stats.replace([np.inf, -np.inf], np.nan)
        .dropna(subset=["relative_speed_loss"])
    )

    if len(speed_stats) < top_k:
        raise RuntimeError(
            f"Fewer than {top_k} valid snow-sensitive stations were found."
        )

    top_stations = (
        speed_stats.sort_values("relative_speed_loss", ascending=False)
        .head(top_k)
        .index
        .tolist()
    )

    selected_data = data[data["station_id"].isin(top_stations)].copy()

    plot_columns = [
        VARIABLE_LABELS["volume"],
        VARIABLE_LABELS["speed"],
        VARIABLE_LABELS["temperature"],
        VARIABLE_LABELS["snow_depth"],
    ]

    full_plot_data = selected_data[plot_columns].dropna().copy()

    if len(full_plot_data) > max_plot_points:
        plot_data = full_plot_data.sample(n=max_plot_points, random_state=random_state)
    else:
        plot_data = full_plot_data.copy()

    snow_speed_data = selected_data[
        [VARIABLE_LABELS["snow_depth"], VARIABLE_LABELS["speed"]]
    ].dropna()

    if len(snow_speed_data) < 3:
        raise RuntimeError(
            "The selected stations have insufficient samples for snow-speed correlation."
        )

    snow_speed_r, snow_speed_p = pearsonr(
        snow_speed_data[VARIABLE_LABELS["snow_depth"]],
        snow_speed_data[VARIABLE_LABELS["speed"]]
    )

    mean_loss_percent = speed_stats.loc[top_stations, "relative_speed_loss"].mean() * 100

    metadata = {
        "resolved_cols": resolved_cols,
        "top_stations": top_stations,
        "mean_loss_percent": mean_loss_percent,
        "selected_records": len(selected_data),
        "plot_records": len(plot_data),
        "snow_speed_sample_size": len(snow_speed_data),
        "snow_speed_r": snow_speed_r,
        "snow_speed_p": snow_speed_p,
        "speed_stats": speed_stats.loc[top_stations, [False, True, "relative_speed_loss"]].copy(),
        "sample_counts": sample_counts.loc[top_stations, [False, True]].copy(),
    }

    return full_plot_data, plot_data, metadata


def correlation_panel(x, y, **kwargs):
    pair_data = pd.DataFrame({"x": x, "y": y}).dropna()
    ax = plt.gca()

    if len(pair_data) < 3:
        ax.annotate(
            "n < 3",
            xy=(0.5, 0.5),
            xycoords=ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="0.15"
        )
        return

    r_value, p_value = pearsonr(pair_data["x"], pair_data["y"])

    ax.set_facecolor(CORRELATION_CMAP((r_value + 1) / 2))
    ax.patch.set_alpha(0.90)

    if p_value < 0.001:
        significance = "***"
    elif p_value < 0.01:
        significance = "**"
    elif p_value < 0.05:
        significance = "*"
    else:
        significance = ""

    ax.annotate(
        f"r = {r_value:.2f}{significance}",
        xy=(0.5, 0.5),
        xycoords=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8,
        fontweight="bold",
        color="0.15"
    )


def apply_plain_format(ax):
    try:
        ax.ticklabel_format(style="plain", axis="both", useOffset=False)
    except Exception:
        pass

    ax.xaxis.get_offset_text().set_visible(False)
    ax.yaxis.get_offset_text().set_visible(False)


def apply_variable_formatter(ax, x_var=None, y_var=None, diagonal=False):
    apply_plain_format(ax)

    if x_var in THOUSAND_VARIABLES:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    else:
        ax.xaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))

    if diagonal:
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    else:
        if y_var in THOUSAND_VARIABLES:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
        else:
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))


def format_pairgrid(grid):
    n_vars = len(grid.x_vars)

    for i in range(n_vars):
        for j in range(n_vars):
            ax = grid.axes[i, j]

            if ax is None:
                continue

            ax.grid(True, linestyle=":", linewidth=0.35, alpha=0.35)

            ax.tick_params(
                axis="both",
                direction="out",
                length=2.5,
                width=0.6,
                labelsize=6
            )

            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(0.8)

            x_var = grid.x_vars[j]
            y_var = grid.y_vars[i]

            apply_variable_formatter(
                ax,
                x_var=x_var,
                y_var=y_var,
                diagonal=(i == j)
            )

            if i < n_vars - 1:
                ax.set_xlabel("")

            if j > 0:
                ax.set_ylabel("")

            ax.xaxis.label.set_size(8)
            ax.yaxis.label.set_size(8)

    for ax in grid.axes[-1, :]:
        ax.set_xlabel(ax.get_xlabel(), fontsize=8)

    for ax in grid.axes[:, 0]:
        ax.set_ylabel(ax.get_ylabel(), fontsize=8)


def plot_snow_sensitive_correlation(
    input_file,
    output_dir,
    output_prefix,
    station_col,
    volume_col,
    speed_col,
    temperature_col,
    snow_depth_col,
    snow_threshold_mm,
    speed_min,
    speed_max,
    min_snow_samples,
    min_clear_samples,
    top_k,
    max_plot_points,
    random_state,
    font_name,
    pair_height
):
    set_publication_style(font_name)
    sns.set_style("white")

    full_plot_data, plot_data, metadata = prepare_snow_sensitive_data(
        input_file=input_file,
        station_col=station_col,
        volume_col=volume_col,
        speed_col=speed_col,
        temperature_col=temperature_col,
        snow_depth_col=snow_depth_col,
        snow_threshold_mm=snow_threshold_mm,
        speed_min=speed_min,
        speed_max=speed_max,
        min_snow_samples=min_snow_samples,
        min_clear_samples=min_clear_samples,
        top_k=top_k,
        max_plot_points=max_plot_points,
        random_state=random_state
    )

    grid = sns.PairGrid(
        plot_data,
        diag_sharey=False,
        height=pair_height,
        aspect=1.0
    )

    grid.map_diag(
        sns.histplot,
        color="#5B8DB8",
        kde=True,
        element="step",
        stat="count",
        linewidth=0.8
    )

    for ax in grid.diag_axes:
        if ax is not None and len(ax.lines) > 0:
            ax.lines[-1].set_color("#2F5D7E")
            ax.lines[-1].set_linewidth(1.2)

    grid.map_lower(
        sns.scatterplot,
        s=6,
        alpha=0.22,
        color="#6C7A89",
        edgecolor=None
    )

    grid.map_upper(correlation_panel)
    format_pairgrid(grid)

    grid.fig.set_facecolor("white")
    grid.fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.98,
        wspace=0.06,
        hspace=0.06
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_tif = output_path / f"{output_prefix}.tif"
    output_png = output_path / f"{output_prefix}_preview.png"
    output_txt = output_path / f"{output_prefix}_summary.txt"

    grid.fig.savefig(
        output_tif,
        dpi=600,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight"
    )

    grid.fig.savefig(
        output_png,
        dpi=300,
        format="png",
        bbox_inches="tight"
    )

    correlation_matrix = full_plot_data.corr(method="pearson")

    speed_stats = metadata["speed_stats"].rename(
        columns={
            False: "mean_speed_non_snow",
            True: "mean_speed_snow",
        }
    )

    sample_counts = metadata["sample_counts"].rename(
        columns={
            False: "non_snow_samples",
            True: "snow_samples",
        }
    )

    station_summary = speed_stats.join(sample_counts, how="left")
    station_summary["relative_speed_loss_percent"] = station_summary["relative_speed_loss"] * 100

    summary_lines = [
        f"Input file: {input_file}",
        f"Resolved station column: {metadata['resolved_cols']['station']}",
        f"Resolved volume column: {metadata['resolved_cols']['volume']}",
        f"Resolved speed column: {metadata['resolved_cols']['speed']}",
        f"Resolved temperature column: {metadata['resolved_cols']['temperature']}",
        f"Resolved snow-depth column: {metadata['resolved_cols']['snow_depth']}",
        f"Snow threshold: {snow_threshold_mm} mm",
        f"Speed filter: ({speed_min}, {speed_max}) km/h",
        f"Minimum snow samples per station: {min_snow_samples}",
        f"Minimum non-snow samples per station: {min_clear_samples}",
        f"Top station count: {top_k}",
        f"Top station IDs: {', '.join(metadata['top_stations'])}",
        f"Mean speed loss: {metadata['mean_loss_percent']:.6f}%",
        f"Selected station-hour records: {metadata['selected_records']:,}",
        f"Snow-speed effective sample size: {metadata['snow_speed_sample_size']:,}",
        f"Snow-speed Pearson r: {metadata['snow_speed_r']:.6f}",
        f"Snow-speed Pearson {format_p_value(metadata['snow_speed_p'])}",
        f"Plot sample size: {metadata['plot_records']:,}",
        "",
        "Top-station summary:",
        station_summary.to_string(),
        "",
        "Pearson correlation matrix based on selected stations:",
        correlation_matrix.to_string(),
        "",
        f"TIFF output: {output_tif}",
        f"PNG preview: {output_png}"
    ]

    output_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    plt.close(grid.fig)

    print("\n".join(summary_lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="figure_snow_sensitive_station_correlation")
    parser.add_argument("--station-col", default=None)
    parser.add_argument("--volume-col", default=None)
    parser.add_argument("--speed-col", default=None)
    parser.add_argument("--temperature-col", default=None)
    parser.add_argument("--snow-depth-col", default=None)
    parser.add_argument("--snow-threshold-mm", type=float, default=20.0)
    parser.add_argument("--speed-min", type=float, default=10.0)
    parser.add_argument("--speed-max", type=float, default=140.0)
    parser.add_argument("--min-snow-samples", type=int, default=5)
    parser.add_argument("--min-clear-samples", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--max-plot-points", type=int, default=5000)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--font-name", default="Arial")
    parser.add_argument("--pair-height", type=float, default=1.60)
    return parser.parse_args()


def main():
    args = parse_args()

    plot_snow_sensitive_correlation(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        station_col=args.station_col,
        volume_col=args.volume_col,
        speed_col=args.speed_col,
        temperature_col=args.temperature_col,
        snow_depth_col=args.snow_depth_col,
        snow_threshold_mm=args.snow_threshold_mm,
        speed_min=args.speed_min,
        speed_max=args.speed_max,
        min_snow_samples=args.min_snow_samples,
        min_clear_samples=args.min_clear_samples,
        top_k=args.top_k,
        max_plot_points=args.max_plot_points,
        random_state=args.random_state,
        font_name=args.font_name,
        pair_height=args.pair_height
    )


if __name__ == "__main__":
    main()

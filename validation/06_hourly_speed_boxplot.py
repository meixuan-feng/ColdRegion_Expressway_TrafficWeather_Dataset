# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


SPEED_COLUMN_CANDIDATES = [
    "traffic_speed_kmh",
    "avg_speed",
    "speed",
    "average_speed",
    "speed_kmh"
]


def set_publication_style(font_name="Arial"):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 9
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 7
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"
    plt.rcParams["axes.unicode_minus"] = False


def detect_speed_column(columns, specified_col=None):
    if specified_col is not None:
        if specified_col not in columns:
            raise KeyError(f"Speed column not found: {specified_col}")
        return specified_col

    for col in SPEED_COLUMN_CANDIDATES:
        if col in columns:
            return col

    raise KeyError(
        "No speed column was detected. "
        f"Available columns are: {list(columns)}"
    )


def load_hourly_speed_data(
    input_file,
    datetime_col,
    speed_col=None,
    min_speed=5.0,
    max_speed=160.0
):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0)
    detected_speed_col = detect_speed_column(header.columns, speed_col)

    if datetime_col not in header.columns:
        raise KeyError(f"Datetime column not found: {datetime_col}")

    data = pd.read_csv(
        input_path,
        usecols=[datetime_col, detected_speed_col],
        low_memory=False
    )

    data[datetime_col] = pd.to_datetime(data[datetime_col], errors="coerce")
    data[detected_speed_col] = pd.to_numeric(data[detected_speed_col], errors="coerce")
    data = data.dropna(subset=[datetime_col, detected_speed_col]).copy()

    data = data[
        (data[detected_speed_col] > min_speed) &
        (data[detected_speed_col] < max_speed)
    ].copy()

    data["hour"] = data[datetime_col].dt.hour

    hourly_data = [
        data.loc[data["hour"] == hour, detected_speed_col].values
        for hour in range(24)
    ]

    hourly_mean = (
        data.groupby("hour")[detected_speed_col]
        .mean()
        .reindex(range(24))
    )

    return data, detected_speed_col, hourly_data, hourly_mean


def style_axis(ax):
    ax.grid(True, axis="y", linestyle=":", linewidth=0.5, alpha=0.6)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def plot_hourly_speed_boxplot(
    input_file,
    output_dir,
    output_prefix,
    datetime_col,
    speed_col,
    min_speed,
    max_speed,
    font_name,
    figure_width,
    figure_height,
    show_fliers
):
    set_publication_style(font_name)

    data, detected_speed_col, hourly_data, hourly_mean = load_hourly_speed_data(
        input_file=input_file,
        datetime_col=datetime_col,
        speed_col=speed_col,
        min_speed=min_speed,
        max_speed=max_speed
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=600
    )

    ax.boxplot(
        hourly_data,
        positions=np.arange(24),
        widths=0.55,
        patch_artist=True,
        showfliers=show_fliers,
        boxprops=dict(
            linewidth=0.8,
            facecolor="white"
        ),
        medianprops=dict(
            linewidth=1.0
        ),
        whiskerprops=dict(
            linewidth=0.8
        ),
        capprops=dict(
            linewidth=0.8
        ),
        flierprops=dict(
            marker="o",
            markersize=0.6,
            markerfacecolor="none",
            markeredgewidth=0.25,
            alpha=0.25
        )
    )

    ax.plot(
        hourly_mean.index,
        hourly_mean.values,
        linewidth=1.0,
        marker="o",
        markersize=2.2,
        label="Mean speed",
        zorder=10
    )

    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Average speed (km/h)")
    ax.set_xlim(-0.8, 23.8)
    ax.set_xticks(np.arange(0, 24, 3))
    ax.set_xticklabels([str(hour) for hour in range(0, 24, 3)])

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=2.0
    )

    style_axis(ax)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.16,
        top=0.96
    )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    output_tif = output_path / f"{output_prefix}.tif"
    output_png = output_path / f"{output_prefix}_preview.png"
    output_txt = output_path / f"{output_prefix}_summary.txt"

    fig.savefig(
        output_tif,
        dpi=600,
        format="tiff",
        pil_kwargs={"compression": "tiff_lzw"},
        bbox_inches="tight"
    )

    fig.savefig(
        output_png,
        dpi=300,
        format="png",
        bbox_inches="tight"
    )

    plt.close(fig)

    summary_lines = [
        f"Input file: {input_file}",
        f"Detected speed column: {detected_speed_col}",
        f"Valid records: {len(data):,}",
        f"Speed range retained: ({min_speed}, {max_speed}) km/h",
        f"Mean speed: {data[detected_speed_col].mean():.6f} km/h",
        f"Median speed: {data[detected_speed_col].median():.6f} km/h",
        f"Minimum speed: {data[detected_speed_col].min():.6f} km/h",
        f"Maximum speed: {data[detected_speed_col].max():.6f} km/h",
        f"TIFF output: {output_tif}",
        f"PNG preview: {output_png}"
    ]

    output_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print("\n".join(summary_lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="figure_hourly_speed_boxplot")
    parser.add_argument("--datetime-col", default="datetime")
    parser.add_argument("--speed-col", default=None)
    parser.add_argument("--min-speed", type=float, default=5.0)
    parser.add_argument("--max-speed", type=float, default=160.0)
    parser.add_argument("--font-name", default="Arial")
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=3.6)
    parser.add_argument("--hide-fliers", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    plot_hourly_speed_boxplot(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        datetime_col=args.datetime_col,
        speed_col=args.speed_col,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        font_name=args.font_name,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        show_fliers=not args.hide_fliers
    )


if __name__ == "__main__":
    main()

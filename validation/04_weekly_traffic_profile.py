# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


COLUMN_CANDIDATES = {
    "datetime": ["datetime", "time", "timestamp", "date_time"],
    "passenger_volume": ["volume_passenger_veh", "passenger_volume", "passenger_flow"],
    "truck_volume": ["volume_truck_veh", "truck_volume", "truck_flow", "freight_volume"],
    "speed": ["traffic_speed_kmh", "avg_speed", "average_speed", "speed_kmh", "speed"],
}


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


def load_weekly_profile(
    input_file,
    datetime_col=None,
    passenger_col=None,
    truck_col=None,
    speed_col=None,
    min_speed=5.0,
    max_speed=150.0
):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0)

    resolved_datetime_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["datetime"],
        specified_col=datetime_col,
        target_name="datetime"
    )

    resolved_passenger_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["passenger_volume"],
        specified_col=passenger_col,
        target_name="passenger volume"
    )

    resolved_truck_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["truck_volume"],
        specified_col=truck_col,
        target_name="truck volume"
    )

    resolved_speed_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["speed"],
        specified_col=speed_col,
        target_name="speed"
    )

    use_cols = [
        resolved_datetime_col,
        resolved_passenger_col,
        resolved_truck_col,
        resolved_speed_col,
    ]

    data = pd.read_csv(input_path, usecols=use_cols, low_memory=False)

    data = data.rename(
        columns={
            resolved_datetime_col: "datetime",
            resolved_passenger_col: "passenger_volume",
            resolved_truck_col: "truck_volume",
            resolved_speed_col: "speed",
        }
    )

    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")
    data = data.dropna(subset=["datetime"]).copy()

    for col in ["passenger_volume", "truck_volume", "speed"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data[
        (data["speed"] > min_speed) &
        (data["speed"] < max_speed)
    ].copy()

    data["datetime_hour"] = data["datetime"].dt.floor("h")

    hourly_network = (
        data.groupby("datetime_hour", as_index=False)
        .agg(
            passenger_volume=("passenger_volume", "sum"),
            truck_volume=("truck_volume", "sum"),
            speed=("speed", "mean")
        )
    )

    hourly_network["day_of_week"] = hourly_network["datetime_hour"].dt.dayofweek
    hourly_network["hour"] = hourly_network["datetime_hour"].dt.hour
    hourly_network["continuous_hour"] = hourly_network["day_of_week"] * 24 + hourly_network["hour"]

    profile = (
        hourly_network.groupby("continuous_hour")[["passenger_volume", "truck_volume", "speed"]]
        .mean()
        .reindex(range(168))
        .reset_index()
        .rename(columns={"index": "continuous_hour"})
    )

    metadata = {
        "datetime_col": resolved_datetime_col,
        "passenger_col": resolved_passenger_col,
        "truck_col": resolved_truck_col,
        "speed_col": resolved_speed_col,
        "valid_records": len(data),
        "hourly_records": len(hourly_network)
    }

    return profile, metadata


def format_xaxis(ax):
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    ax.xaxis.set_major_locator(ticker.MultipleLocator(12))

    def formatter(x, pos):
        if x < 0 or x > 167:
            return ""

        hour_index = int(round(x))
        day_idx = hour_index // 24
        hour = hour_index % 24

        if day_idx >= 7:
            return ""

        if hour == 0:
            return f"{days[day_idx]}\n00:00"

        if hour == 12:
            return "12:00"

        return ""

    ax.xaxis.set_major_formatter(ticker.FuncFormatter(formatter))


def style_axis(ax):
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def add_panel_label(ax, label):
    ax.text(
        -0.065,
        1.045,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left"
    )


def plot_weekly_profile(
    input_file,
    output_dir,
    output_prefix,
    datetime_col,
    passenger_col,
    truck_col,
    speed_col,
    min_speed,
    max_speed,
    font_name,
    figure_width,
    figure_height
):
    set_publication_style(font_name)

    profile, metadata = load_weekly_profile(
        input_file=input_file,
        datetime_col=datetime_col,
        passenger_col=passenger_col,
        truck_col=truck_col,
        speed_col=speed_col,
        min_speed=min_speed,
        max_speed=max_speed
    )

    fig, (ax1, ax2) = plt.subplots(
        2,
        1,
        figsize=(figure_width, figure_height),
        dpi=600,
        sharex=False
    )

    ax1.axvspan(120, 168, color="0.85", alpha=0.45, zorder=0)
    ax2.axvspan(120, 168, color="0.85", alpha=0.45, zorder=0)

    ax1.plot(
        profile["continuous_hour"],
        profile["passenger_volume"],
        color="#2ecc71",
        linewidth=1.2,
        label="Passenger vehicles"
    )

    ax1.plot(
        profile["continuous_hour"],
        profile["truck_volume"],
        color="#f1c40f",
        linewidth=1.2,
        label="Trucks"
    )

    ax1.fill_between(
        profile["continuous_hour"],
        profile["passenger_volume"],
        color="#2ecc71",
        alpha=0.10
    )

    ax1.set_ylabel("Average total volume (veh/h)")
    ax1.set_ylim(bottom=0)
    ax1.set_xlim(0, 167)
    ax1.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    weekend_patch = mpatches.Patch(
        facecolor="0.85",
        edgecolor="none",
        alpha=0.45,
        label="Weekend"
    )

    handles, _ = ax1.get_legend_handles_labels()
    handles.append(weekend_patch)

    ax1.legend(
        handles=handles,
        loc="upper right",
        frameon=False,
        handlelength=2.0
    )

    format_xaxis(ax1)
    style_axis(ax1)
    ax1.tick_params(labelbottom=True)
    add_panel_label(ax1, "a")

    ax2.plot(
        profile["continuous_hour"],
        profile["speed"],
        color="#34495e",
        linewidth=1.2,
        label="Average speed"
    )

    mean_speed = profile["speed"].mean()

    ax2.axhline(
        y=mean_speed,
        color="#c0392b",
        linestyle="--",
        linewidth=1.0,
        label=f"Weekly mean = {mean_speed:.1f} km/h"
    )

    ax2.set_ylabel("Average speed (km/h)")
    ax2.set_xlabel("Time of week")
    ax2.set_xlim(0, 167)

    ax2.legend(
        loc="lower right",
        frameon=False,
        handlelength=2.0
    )

    format_xaxis(ax2)
    style_axis(ax2)
    add_panel_label(ax2, "b")

    fig.subplots_adjust(
        left=0.12,
        right=0.98,
        bottom=0.12,
        top=0.97,
        hspace=0.34
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
        f"Detected datetime column: {metadata['datetime_col']}",
        f"Detected passenger-volume column: {metadata['passenger_col']}",
        f"Detected truck-volume column: {metadata['truck_col']}",
        f"Detected speed column: {metadata['speed_col']}",
        f"Valid station-hour records: {metadata['valid_records']:,}",
        f"Aggregated network-hour records: {metadata['hourly_records']:,}",
        f"Mean weekly speed: {mean_speed:.6f} km/h",
        f"TIFF output: {output_tif}",
        f"PNG preview: {output_png}"
    ]

    output_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print("\n".join(summary_lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="figure_weekly_traffic_profile")
    parser.add_argument("--datetime-col", default=None)
    parser.add_argument("--passenger-col", default=None)
    parser.add_argument("--truck-col", default=None)
    parser.add_argument("--speed-col", default=None)
    parser.add_argument("--min-speed", type=float, default=5.0)
    parser.add_argument("--max-speed", type=float, default=150.0)
    parser.add_argument("--font-name", default="Arial")
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=4.8)
    return parser.parse_args()


def main():
    args = parse_args()

    plot_weekly_profile(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        datetime_col=args.datetime_col,
        passenger_col=args.passenger_col,
        truck_col=args.truck_col,
        speed_col=args.speed_col,
        min_speed=args.min_speed,
        max_speed=args.max_speed,
        font_name=args.font_name,
        figure_width=args.figure_width,
        figure_height=args.figure_height
    )


if __name__ == "__main__":
    main()

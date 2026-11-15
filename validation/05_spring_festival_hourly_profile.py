# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter


COLUMN_CANDIDATES = {
    "station": ["station_id", "raw_station_code", "station_code"],
    "datetime": ["datetime", "time", "timestamp", "date_time"],
    "total_volume": ["volume_total_veh", "total_volume", "volume_total", "total_flow"],
    "passenger_volume": ["volume_passenger_veh", "passenger_volume", "passenger_flow"],
    "truck_volume": ["volume_truck_veh", "truck_volume", "truck_flow", "freight_volume"],
}


CNY_DATES = {
    2020: "2020-01-25",
    2021: "2021-02-12",
    2022: "2022-02-01",
    2023: "2023-01-22",
    2024: "2024-02-10",
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


def detect_cny_date(datetime_min, datetime_max, specified_cny_date=None):
    if specified_cny_date is not None:
        return pd.Timestamp(specified_cny_date)

    for cny_date_string in CNY_DATES.values():
        cny_date = pd.Timestamp(cny_date_string)

        if datetime_min <= cny_date - pd.Timedelta(days=5) and datetime_max >= cny_date + pd.Timedelta(days=10):
            return cny_date

    raise ValueError(
        "No complete Spring Festival period was detected. "
        "Please specify --cny-date manually, for example --cny-date 2023-01-22."
    )


def load_spring_festival_profile(
    input_file,
    station_col=None,
    datetime_col=None,
    total_volume_col=None,
    passenger_col=None,
    truck_col=None,
    top_n=5,
    cny_date=None,
    before_days=3,
    after_days=9
):
    input_path = Path(input_file)

    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    header = pd.read_csv(input_path, nrows=0)

    resolved_station_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["station"],
        specified_col=station_col,
        target_name="station"
    )

    resolved_datetime_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["datetime"],
        specified_col=datetime_col,
        target_name="datetime"
    )

    resolved_total_col = resolve_column(
        header.columns,
        COLUMN_CANDIDATES["total_volume"],
        specified_col=total_volume_col,
        target_name="total volume"
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

    use_cols = [
        resolved_station_col,
        resolved_datetime_col,
        resolved_total_col,
        resolved_passenger_col,
        resolved_truck_col,
    ]

    data = pd.read_csv(input_path, usecols=use_cols, low_memory=False)

    data = data.rename(
        columns={
            resolved_station_col: "station_id",
            resolved_datetime_col: "datetime",
            resolved_total_col: "total_volume",
            resolved_passenger_col: "passenger_volume",
            resolved_truck_col: "truck_volume",
        }
    )

    data["station_id"] = data["station_id"].astype(str).str.strip()
    data["datetime"] = pd.to_datetime(data["datetime"], errors="coerce")

    for col in ["total_volume", "passenger_volume", "truck_volume"]:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    data = data.dropna(subset=["station_id", "datetime", "total_volume"]).copy()

    top_stations = (
        data.groupby("station_id")["total_volume"]
        .sum()
        .sort_values(ascending=False)
        .head(top_n)
        .index
        .tolist()
    )

    top_data = data[data["station_id"].isin(top_stations)].copy()
    top_data = top_data.dropna(subset=["passenger_volume", "truck_volume"]).copy()

    selected_cny_date = detect_cny_date(
        datetime_min=top_data["datetime"].min(),
        datetime_max=top_data["datetime"].max(),
        specified_cny_date=cny_date
    )

    start_time = selected_cny_date - pd.Timedelta(days=before_days)
    end_time = selected_cny_date + pd.Timedelta(days=after_days) - pd.Timedelta(hours=1)

    cny_data = top_data[
        (top_data["datetime"] >= start_time) &
        (top_data["datetime"] <= end_time)
    ].copy()

    hourly_profile = (
        cny_data.groupby("datetime", as_index=False)[["passenger_volume", "truck_volume"]]
        .mean()
        .sort_values("datetime")
    )

    if hourly_profile.empty:
        raise ValueError("The Spring Festival window contains no valid records.")

    hourly_profile["days_to_cny"] = (
        hourly_profile["datetime"] - selected_cny_date
    ).dt.total_seconds() / 86400

    metadata = {
        "station_col": resolved_station_col,
        "datetime_col": resolved_datetime_col,
        "total_volume_col": resolved_total_col,
        "passenger_col": resolved_passenger_col,
        "truck_col": resolved_truck_col,
        "top_stations": top_stations,
        "cny_date": selected_cny_date,
        "start_time": start_time,
        "end_time": end_time,
        "valid_records": len(cny_data),
        "hourly_records": len(hourly_profile),
    }

    return hourly_profile, metadata


def style_axis(ax):
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ax.tick_params(
        axis="both",
        direction="out",
        length=3,
        width=0.8
    )

    ax.tick_params(
        axis="x",
        which="minor",
        direction="out",
        length=2,
        width=0.6
    )

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def annotate_spring_festival_features(ax, profile, cny_date):
    eve_date = cny_date - pd.Timedelta(days=1)
    eve_data = profile[
        (profile["datetime"].dt.date == eve_date.date()) &
        (profile["datetime"].dt.hour > 18)
    ]

    if not eve_data.empty:
        min_row = eve_data.loc[eve_data["passenger_volume"].idxmin()]

        ax.annotate(
            "CNY eve",
            xy=(min_row["datetime"], min_row["passenger_volume"]),
            xytext=(0, 25),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            arrowprops=dict(
                arrowstyle="->",
                color="black",
                linewidth=0.7
            )
        )

    return_rush_date = cny_date + pd.Timedelta(days=6)
    return_rush_data = profile[profile["datetime"].dt.date == return_rush_date.date()]

    if not return_rush_data.empty:
        max_row = return_rush_data.loc[return_rush_data["passenger_volume"].idxmax()]

        ax.annotate(
            "Return rush peak",
            xy=(max_row["datetime"], max_row["passenger_volume"]),
            xytext=(-8, 22),
            textcoords="offset points",
            ha="right",
            fontsize=7,
            color="#c0392b",
            arrowprops=dict(
                arrowstyle="->",
                color="#c0392b",
                linewidth=0.7
            )
        )


def plot_spring_festival_profile(
    input_file,
    output_dir,
    output_prefix,
    station_col,
    datetime_col,
    total_volume_col,
    passenger_col,
    truck_col,
    top_n,
    cny_date,
    before_days,
    after_days,
    font_name,
    figure_width,
    figure_height
):
    set_publication_style(font_name)

    profile, metadata = load_spring_festival_profile(
        input_file=input_file,
        station_col=station_col,
        datetime_col=datetime_col,
        total_volume_col=total_volume_col,
        passenger_col=passenger_col,
        truck_col=truck_col,
        top_n=top_n,
        cny_date=cny_date,
        before_days=before_days,
        after_days=after_days
    )

    fig, ax = plt.subplots(
        figsize=(figure_width, figure_height),
        dpi=600
    )

    holiday_start = metadata["cny_date"] - pd.Timedelta(days=1)
    holiday_end = metadata["cny_date"] + pd.Timedelta(days=7)

    ax.axvspan(
        holiday_start,
        holiday_end,
        color="0.85",
        alpha=0.45,
        zorder=0
    )

    ax.plot(
        profile["datetime"],
        profile["passenger_volume"],
        color="#2ecc71",
        linewidth=1.2,
        label="Passenger vehicles"
    )

    ax.plot(
        profile["datetime"],
        profile["truck_volume"],
        color="#f1c40f",
        linewidth=1.2,
        label="Trucks"
    )

    ax.fill_between(
        profile["datetime"],
        profile["passenger_volume"],
        color="#2ecc71",
        alpha=0.10
    )

    annotate_spring_festival_features(ax, profile, metadata["cny_date"])

    ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax.xaxis.set_minor_locator(mdates.HourLocator(interval=6))

    ax.set_ylabel("Average volume (veh/h)")
    ax.set_xlabel("Date")
    ax.set_xlim(profile["datetime"].min(), profile["datetime"].max())
    ax.set_ylim(bottom=0)
    ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))

    holiday_patch = mpatches.Patch(
        facecolor="0.85",
        edgecolor="none",
        alpha=0.45,
        label="Holiday period"
    )

    handles, _ = ax.get_legend_handles_labels()
    handles.append(holiday_patch)

    ax.legend(
        handles=handles,
        loc="upper left",
        frameon=False,
        handlelength=2.0
    )

    style_axis(ax)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        bottom=0.18,
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
        f"Detected station column: {metadata['station_col']}",
        f"Detected datetime column: {metadata['datetime_col']}",
        f"Detected total-volume column: {metadata['total_volume_col']}",
        f"Detected passenger-volume column: {metadata['passenger_col']}",
        f"Detected truck-volume column: {metadata['truck_col']}",
        f"Spring Festival date: {metadata['cny_date'].date()}",
        f"Window start: {metadata['start_time']}",
        f"Window end: {metadata['end_time']}",
        f"Top station count: {top_n}",
        f"Top stations: {', '.join(metadata['top_stations'])}",
        f"Valid station-hour records in window: {metadata['valid_records']:,}",
        f"Hourly profile records: {metadata['hourly_records']:,}",
        f"TIFF output: {output_tif}",
        f"PNG preview: {output_png}"
    ]

    output_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print("\n".join(summary_lines))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="figure_spring_festival_hourly_profile")
    parser.add_argument("--station-col", default=None)
    parser.add_argument("--datetime-col", default=None)
    parser.add_argument("--total-volume-col", default=None)
    parser.add_argument("--passenger-col", default=None)
    parser.add_argument("--truck-col", default=None)
    parser.add_argument("--top-n", type=int, default=5)
    parser.add_argument("--cny-date", default=None)
    parser.add_argument("--before-days", type=int, default=3)
    parser.add_argument("--after-days", type=int, default=9)
    parser.add_argument("--font-name", default="Arial")
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=3.6)
    return parser.parse_args()


def main():
    args = parse_args()

    plot_spring_festival_profile(
        input_file=args.input_file,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        station_col=args.station_col,
        datetime_col=args.datetime_col,
        total_volume_col=args.total_volume_col,
        passenger_col=args.passenger_col,
        truck_col=args.truck_col,
        top_n=args.top_n,
        cny_date=args.cny_date,
        before_days=args.before_days,
        after_days=args.after_days,
        font_name=args.font_name,
        figure_width=args.figure_width,
        figure_height=args.figure_height
    )


if __name__ == "__main__":
    main()

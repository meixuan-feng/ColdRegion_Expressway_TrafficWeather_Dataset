# -*- coding: utf-8 -*-

import argparse
import glob
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import geopandas as gpd
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Polygon

warnings.filterwarnings("ignore")

MAIN_CRS = ccrs.PlateCarree()


def set_publication_style(font_name="Arial"):
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = [font_name, "Helvetica", "Arial", "DejaVu Sans"]
    plt.rcParams["font.size"] = 8
    plt.rcParams["axes.labelsize"] = 8
    plt.rcParams["xtick.labelsize"] = 8
    plt.rcParams["ytick.labelsize"] = 8
    plt.rcParams["legend.fontsize"] = 6
    plt.rcParams["axes.linewidth"] = 0.8
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["axes.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["savefig.edgecolor"] = "white"
    plt.rcParams["axes.unicode_minus"] = False


def normalize_id(series):
    return series.astype(str).str.strip().str.replace(r"\.0$", "", regex=True)


def add_panel_label(ax, label):
    ax.text(
        0.025,
        0.975,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
        zorder=40,
        bbox=dict(
            facecolor="white",
            edgecolor="none",
            alpha=0.85,
            pad=1.8
        )
    )


def add_north_arrow(ax, x=0.925, y=0.775, size=0.045):
    ax.add_patch(
        Polygon(
            [[x, y + size], [x - size * 0.25, y], [x, y]],
            color="black",
            transform=ax.transAxes,
            zorder=35
        )
    )

    ax.add_patch(
        Polygon(
            [[x, y + size], [x + size * 0.25, y], [x, y]],
            color="white",
            ec="black",
            lw=0.45,
            transform=ax.transAxes,
            zorder=35
        )
    )

    ax.text(
        x,
        y + size + 0.010,
        "N",
        ha="center",
        va="bottom",
        fontsize=6.5,
        fontweight="bold",
        transform=ax.transAxes,
        zorder=35
    )


def add_manual_scale_bar(ax, map_extent, length_km=200, x_frac=0.72, y_frac=0.955, linewidth=1.2):
    lon_min, lon_max, lat_min, lat_max = map_extent

    lat_center = (lat_min + lat_max) / 2.0
    km_per_degree_lon = 111.32 * np.cos(np.deg2rad(lat_center))
    length_degree = length_km / km_per_degree_lon

    x0 = lon_min + (lon_max - lon_min) * x_frac
    y0 = lat_min + (lat_max - lat_min) * y_frac

    if x0 + length_degree > lon_max - 0.15:
        x0 = lon_max - length_degree - 0.25

    x1 = x0 + length_degree
    tick_height = (lat_max - lat_min) * 0.012

    ax.plot(
        [x0, x1],
        [y0, y0],
        color="black",
        linewidth=linewidth,
        transform=MAIN_CRS,
        zorder=35
    )

    ax.plot(
        [x0, x0],
        [y0 - tick_height / 2, y0 + tick_height / 2],
        color="black",
        linewidth=linewidth,
        transform=MAIN_CRS,
        zorder=35
    )

    ax.plot(
        [x1, x1],
        [y0 - tick_height / 2, y0 + tick_height / 2],
        color="black",
        linewidth=linewidth,
        transform=MAIN_CRS,
        zorder=35
    )

    ax.text(
        (x0 + x1) / 2,
        y0 - (lat_max - lat_min) * 0.028,
        f"{length_km} km",
        ha="center",
        va="top",
        fontsize=6.5,
        color="black",
        transform=MAIN_CRS,
        zorder=35
    )


def add_decorations(ax, map_extent, scale_length_km):
    add_manual_scale_bar(ax, map_extent=map_extent, length_km=scale_length_km)
    add_north_arrow(ax)

    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)


def draw_base_map(ax, admin_shp, road_shp_dir, map_extent, show_admin_legend=False):
    admin_path = Path(admin_shp)
    road_dir = Path(road_shp_dir)

    if admin_path.exists():
        gdf_admin = gpd.read_file(admin_path).to_crs("EPSG:4326")
        color_col = next(
            (
                col for col in ["市名", "Name", "NAME", "name", "City"]
                if col in gdf_admin.columns
            ),
            None
        )

        if color_col:
            gdf_admin.plot(
                ax=ax,
                column=color_col,
                cmap="Pastel1",
                alpha=0.30,
                edgecolor="0.35",
                linewidth=0.50,
                zorder=1,
                transform=MAIN_CRS
            )
        else:
            gdf_admin.plot(
                ax=ax,
                color="0.90",
                alpha=0.30,
                edgecolor="0.35",
                linewidth=0.50,
                zorder=1,
                transform=MAIN_CRS
            )

    if road_dir.exists():
        shp_files = glob.glob(str(road_dir / "*.shp"))

        if len(shp_files) > 0:
            gdf_roads = gpd.read_file(shp_files[0]).to_crs("EPSG:4326")
            ax.add_geometries(
                gdf_roads.geometry,
                crs=MAIN_CRS,
                facecolor="none",
                edgecolor="#ff8c00",
                linewidth=0.55,
                alpha=0.90,
                zorder=2
            )

    ax.set_extent(map_extent, crs=MAIN_CRS)
    ax.set_xticks([])
    ax.set_yticks([])


def load_station_coordinates(coordinate_file, station_col, lon_col, lat_col):
    coordinate_path = Path(coordinate_file)

    if not coordinate_path.exists():
        raise FileNotFoundError(f"Coordinate file not found: {coordinate_path}")

    if coordinate_path.suffix.lower() in [".xlsx", ".xls"]:
        coordinates = pd.read_excel(coordinate_path, dtype=str)
    else:
        coordinates = pd.read_csv(coordinate_path, dtype=str, low_memory=False)

    required_cols = [station_col, lon_col, lat_col]
    missing_cols = [col for col in required_cols if col not in coordinates.columns]

    if missing_cols:
        raise KeyError(f"Missing coordinate columns: {missing_cols}")

    coordinates = coordinates[required_cols].copy()
    coordinates = coordinates.rename(
        columns={
            station_col: "station_id",
            lon_col: "lon",
            lat_col: "lat"
        }
    )

    coordinates["station_id"] = normalize_id(coordinates["station_id"])
    coordinates["lon"] = pd.to_numeric(coordinates["lon"], errors="coerce")
    coordinates["lat"] = pd.to_numeric(coordinates["lat"], errors="coerce")

    coordinates = coordinates.dropna(subset=["station_id", "lon", "lat"])
    coordinates = coordinates.drop_duplicates(subset=["station_id"])

    return coordinates


def assign_quantile_grades(values, q=5):
    grades, bins = pd.qcut(
        values,
        q=q,
        labels=False,
        retbins=True,
        duplicates="drop"
    )

    return grades, bins


def prepare_station_metric_data(data_file, coordinates, value_col, metric):
    data_path = Path(data_file)

    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")

    data = pd.read_csv(data_path, low_memory=False)

    required_cols = ["station_id", value_col]
    missing_cols = [col for col in required_cols if col not in data.columns]

    if missing_cols:
        raise KeyError(f"Missing data columns: {missing_cols}")

    data["station_id"] = normalize_id(data["station_id"])
    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.dropna(subset=[value_col]).copy()

    if metric == "sum":
        station_values = (
            data.groupby("station_id", as_index=False)[value_col]
            .sum()
            .rename(columns={value_col: "value"})
        )
    elif metric == "mean":
        station_values = (
            data.groupby("station_id", as_index=False)[value_col]
            .mean()
            .rename(columns={value_col: "value"})
        )
    else:
        raise ValueError("metric must be either 'sum' or 'mean'.")

    merged = station_values.merge(coordinates, on="station_id", how="inner")
    merged["grade"], bins = assign_quantile_grades(merged["value"], q=5)
    merged = merged.dropna(subset=["grade"]).copy()
    merged["grade"] = merged["grade"].astype(int)

    return merged, bins


def build_legend_items(data, bins, colors, point_sizes, value_type, show_admin_legend):
    legend_items = []

    for grade in sorted(data["grade"].unique()):
        idx = int(grade)

        if idx >= len(colors) or idx + 1 >= len(bins):
            continue

        if value_type == "volume":
            low, high = int(bins[idx]), int(bins[idx + 1])
            label = f"{low:,}–{high:,}"
        else:
            low, high = float(bins[idx]), float(bins[idx + 1])
            label = f"{low:.1f}–{high:.1f}"

        legend_items.append(
            mlines.Line2D(
                [],
                [],
                color="white",
                marker="o",
                markerfacecolor=colors[idx],
                markeredgecolor="0.20",
                markeredgewidth=0.35,
                markersize=np.sqrt(point_sizes[idx]) * 0.95,
                label=label,
                linestyle="None"
            )
        )

    legend_items.append(mlines.Line2D([], [], color="none", label=" "))

    legend_items.append(
        mlines.Line2D(
            [],
            [],
            color="#ff8c00",
            linewidth=0.85,
            label="Expressway network"
        )
    )

    if show_admin_legend:
        legend_items.append(
            mpatches.Patch(
                facecolor="0.88",
                edgecolor="0.35",
                alpha=0.5,
                label="Administrative districts"
            )
        )

    return legend_items


def scatter_classified_points(ax, data, colors, point_sizes):
    for grade in sorted(data["grade"].unique()):
        idx = int(grade)
        subset = data[data["grade"] == idx]

        ax.scatter(
            subset["lon"],
            subset["lat"],
            c=colors[idx],
            s=point_sizes[idx],
            marker="o",
            edgecolors="0.20",
            linewidth=0.30,
            zorder=10,
            transform=MAIN_CRS
        )


def add_compact_legend(ax, handles, title):
    legend = ax.legend(
        handles=handles,
        loc="lower left",
        bbox_to_anchor=(0.020, 0.040),
        title=title,
        fontsize=5.7,
        title_fontsize=6.2,
        frameon=True,
        edgecolor="0.45",
        framealpha=0.90,
        borderpad=0.32,
        labelspacing=0.22,
        handlelength=1.00,
        handletextpad=0.32,
        borderaxespad=0.20,
        columnspacing=0.40
    )

    legend.set_zorder(35)
    legend.get_frame().set_linewidth(0.55)

    return legend


def plot_spatial_distribution(
    data_file,
    coordinate_file,
    admin_shp,
    road_shp_dir,
    output_dir,
    output_prefix,
    speed_col,
    volume_col,
    station_col,
    lon_col,
    lat_col,
    extent,
    font_name,
    figure_width,
    figure_height,
    scale_length_km,
    show_admin_legend
):
    set_publication_style(font_name)

    coordinates = load_station_coordinates(
        coordinate_file=coordinate_file,
        station_col=station_col,
        lon_col=lon_col,
        lat_col=lat_col
    )

    volume_data, volume_bins = prepare_station_metric_data(
        data_file=data_file,
        coordinates=coordinates,
        value_col=volume_col,
        metric="sum"
    )

    speed_data, speed_bins = prepare_station_metric_data(
        data_file=data_file,
        coordinates=coordinates,
        value_col=speed_col,
        metric="mean"
    )

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(figure_width, figure_height),
        dpi=600,
        subplot_kw={"projection": MAIN_CRS}
    )

    point_sizes = [10, 15, 20, 25, 30]

    volume_colors = [
        "#fee5d9",
        "#fcae91",
        "#fb6a4a",
        "#de2d26",
        "#a50f15"
    ]

    speed_colors = [
        "#e5f5e0",
        "#c7e9c0",
        "#74c476",
        "#31a354",
        "#006d2c"
    ]

    draw_base_map(
        ax=ax1,
        admin_shp=admin_shp,
        road_shp_dir=road_shp_dir,
        map_extent=extent,
        show_admin_legend=show_admin_legend
    )

    scatter_classified_points(ax1, volume_data, volume_colors, point_sizes)

    volume_legend_items = build_legend_items(
        data=volume_data,
        bins=volume_bins,
        colors=volume_colors,
        point_sizes=point_sizes,
        value_type="volume",
        show_admin_legend=show_admin_legend
    )

    add_compact_legend(
        ax=ax1,
        handles=volume_legend_items,
        title="Total traffic volume\n(vehicles)"
    )

    add_decorations(ax1, map_extent=extent, scale_length_km=scale_length_km)
    add_panel_label(ax1, "a")

    draw_base_map(
        ax=ax2,
        admin_shp=admin_shp,
        road_shp_dir=road_shp_dir,
        map_extent=extent,
        show_admin_legend=show_admin_legend
    )

    scatter_classified_points(ax2, speed_data, speed_colors, point_sizes)

    speed_legend_items = build_legend_items(
        data=speed_data,
        bins=speed_bins,
        colors=speed_colors,
        point_sizes=point_sizes,
        value_type="speed",
        show_admin_legend=show_admin_legend
    )

    add_compact_legend(
        ax=ax2,
        handles=speed_legend_items,
        title="Average speed\n(km/h)"
    )

    add_decorations(ax2, map_extent=extent, scale_length_km=scale_length_km)
    add_panel_label(ax2, "b")

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        bottom=0.045,
        top=0.98,
        wspace=0.035
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
        bbox_inches="tight",
        pad_inches=0.03
    )

    fig.savefig(
        output_png,
        dpi=300,
        format="png",
        bbox_inches="tight",
        pad_inches=0.03
    )

    plt.close(fig)

    summary_lines = [
        f"Data file: {data_file}",
        f"Coordinate file: {coordinate_file}",
        f"Administrative shapefile: {admin_shp}",
        f"Road shapefile directory: {road_shp_dir}",
        f"Volume stations: {len(volume_data):,}",
        f"Speed stations: {len(speed_data):,}",
        f"TIFF output: {output_tif}",
        f"PNG preview: {output_png}"
    ]

    output_txt.write_text("\n".join(summary_lines), encoding="utf-8-sig")

    print("\n".join(summary_lines))


def parse_extent(values):
    if len(values) != 4:
        raise ValueError("--extent must contain four values: lon_min lon_max lat_min lat_max")
    return [float(v) for v in values]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-file", required=True)
    parser.add_argument("--coordinate-file", required=True)
    parser.add_argument("--admin-shp", required=True)
    parser.add_argument("--road-shp-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-prefix", default="figure_spatial_distribution")
    parser.add_argument("--speed-col", default="avg_speed")
    parser.add_argument("--volume-col", default="total_volume")
    parser.add_argument("--station-col", default="station_id")
    parser.add_argument("--lon-col", default="lon")
    parser.add_argument("--lat-col", default="lat")
    parser.add_argument("--extent", nargs=4, default=[120.5, 132.5, 39.5, 47.5])
    parser.add_argument("--font-name", default="Arial")
    parser.add_argument("--figure-width", type=float, default=7.2)
    parser.add_argument("--figure-height", type=float, default=3.8)
    parser.add_argument("--scale-length-km", type=float, default=200)
    parser.add_argument("--show-admin-legend", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    plot_spatial_distribution(
        data_file=args.data_file,
        coordinate_file=args.coordinate_file,
        admin_shp=args.admin_shp,
        road_shp_dir=args.road_shp_dir,
        output_dir=args.output_dir,
        output_prefix=args.output_prefix,
        speed_col=args.speed_col,
        volume_col=args.volume_col,
        station_col=args.station_col,
        lon_col=args.lon_col,
        lat_col=args.lat_col,
        extent=parse_extent(args.extent),
        font_name=args.font_name,
        figure_width=args.figure_width,
        figure_height=args.figure_height,
        scale_length_km=args.scale_length_km,
        show_admin_legend=args.show_admin_legend
    )


if __name__ == "__main__":
    main()

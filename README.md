# Cold-Region Expressway Traffic–Weather Dataset: Source Code

This repository contains the Python scripts used to process toll-based traffic records, inspect and prepare ERA5 weather data, construct the released station-hour traffic–weather dataset, and reproduce the dataset's technical-validation figures.

The associated study describes an hourly traffic–weather dataset for 211 expressway toll stations in Jilin Province, China, covering 1 November 2022 to 31 March 2023.

## Repository structure

```text
.
├── README.md
├── LICENSE
├── requirements.txt
├── src/
│   ├── 01_aggregate_toll_records_hourly.py
│   ├── 02_match_station_coordinates.py
│   ├── 03_inspect_era5_netcdf_structure.py
│   ├── 04_update_snow_depth_from_fusion.py
│   └── 05_build_publication_dataset.py
└── validation/
    ├── 01_data_completeness_heatmap.py
    ├── 02_speed_distribution_diurnal_variation.py
    ├── 03_spatial_distribution_map.py
    ├── 04_weekly_traffic_profile.py
    ├── 05_spring_festival_hourly_profile.py
    ├── 06_hourly_speed_boxplot.py
    ├── 07_networkwide_traffic_weather_correlation.py
    └── 08_snow_sensitive_station_correlation.py
```

## Data sources and access

### Toll transaction data

The original vehicle-level toll transaction records were collected by Jilin Expressway Group and managed and provided by the Comprehensive Planning Division of the Jilin Provincial Department of Transport. These records are not included in this repository because they are not publicly downloadable. Access is subject to approval by the relevant institutions and applicable data-security requirements. Researchers seeking access should contact the Comprehensive Planning Division through the official website of the Jilin Provincial Department of Transport: <https://jtyst.jl.gov.cn/>.

The traffic-processing scripts expect the source fields used in the project, including the exit-station identifier, vehicle-type code, entry and exit timestamps, and trip distance. For the aggregation script, the original CSV files use the field names `出口站编号`, `最终车型`, `出口日期时间`, `入口日期时间`, and `实际里程`.

### ERA5 weather data

The weather data used in the study are from the ERA5 hourly data on single levels from 1940 to present dataset, distributed by the Copernicus Climate Change Service (C3S) Climate Data Store (CDS): <https://doi.org/10.24381/cds.adbb2d47>. The study uses the `2m temperature` and `Snow depth` variables. The original ERA5 files are not included in this code repository and can be retrieved from the CDS using the dataset identifier, variables, dates, and geographical selection described in the associated article.

### Processed dataset

The processed, anonymized, and aggregated station-hour traffic–weather dataset is a separate research output and is distributed separately from this source-code repository. The original vehicle-level toll records, internal station mappings, and other restricted input files are not included here. For each station-hour, total traffic volume equals the sum of passenger-vehicle, truck, and other-vehicle volumes. Vehicle types 1–4 are classified as passenger vehicles, types 11 and above as trucks, and all remaining or missing vehicle-type codes as other vehicles.

## Software environment

The scripts were developed for Python 3.11. Install the listed dependencies from the repository root:

```bash
python -m pip install -r requirements.txt
```

The dependencies include pandas, NumPy, Matplotlib, Seaborn, SciPy, GeoPandas, Cartopy, openpyxl, netCDF4, and xarray. Some geospatial packages may require platform-specific system libraries. A Conda environment using the `conda-forge` channel may be helpful when installing GeoPandas or Cartopy.

To view the available command-line options for an individual script, run it with `--help`. For example:

```bash
python src/01_aggregate_toll_records_hourly.py --help
python validation/01_data_completeness_heatmap.py --help
```

All paths to input and output files are supplied by command-line arguments. Create the required local input and output directories before running the scripts.

## Source scripts

### Data processing (`src/`)

| Script | Description |
|---|---|
| `01_aggregate_toll_records_hourly.py` | Reads monthly toll-transaction CSV files, applies record and trip-speed filters, and aggregates total, passenger, truck, and other-vehicle trips by exit station and exit hour. |
| `02_match_station_coordinates.py` | Matches station-level traffic records with station-coordinate information and writes a report of unmatched stations. |
| `03_inspect_era5_netcdf_structure.py` | Reports the dimensions, coordinates, variables, and time information in an ERA5 NetCDF file. |
| `04_update_snow_depth_from_fusion.py` | Updates the snow-depth field in a station-hour table using a weather-fusion output and a station-ID mapping file. |
| `05_build_publication_dataset.py` | Builds the publication data package, including the hourly traffic–weather table, station-location table, variable-description table, and a summary report; it also checks hourly vehicle-volume consistency. |

### Technical validation and figure generation (`validation/`)

| Script | Description |
|---|---|
| `01_data_completeness_heatmap.py` | Generates the station–date data-completeness heatmap. |
| `02_speed_distribution_diurnal_variation.py` | Generates the traffic-speed distribution and diurnal-variation figures. |
| `03_spatial_distribution_map.py` | Generates spatial maps of cumulative traffic volume and average traffic speed. Requires administrative-boundary and expressway-network GIS shapefiles. |
| `04_weekly_traffic_profile.py` | Generates weekly profiles of passenger-vehicle volume, truck volume, and traffic speed. |
| `05_spring_festival_hourly_profile.py` | Generates hourly traffic profiles for the selected high-volume stations around the Spring Festival. |
| `06_hourly_speed_boxplot.py` | Generates hourly traffic-speed boxplots. |
| `07_networkwide_traffic_weather_correlation.py` | Generates the network-wide traffic–weather correlation analysis and figure. |
| `08_snow_sensitive_station_correlation.py` | Identifies snow-sensitive stations and generates the corresponding traffic–weather correlation analysis and figure. |

## Running the scripts

The scripts are independent command-line programs. Run each script from the repository root and provide the input and output paths required by that script. Use `--help` to see all required and optional arguments.

The main processing sequence is:

1. Aggregate the original toll transaction records using `src/01_aggregate_toll_records_hourly.py`.
2. Match station coordinates using `src/02_match_station_coordinates.py`.
3. Inspect the downloaded ERA5 NetCDF files using `src/03_inspect_era5_netcdf_structure.py`.
4. Where required by the project workflow, update snow-depth values using `src/04_update_snow_depth_from_fusion.py`.
5. Construct the publication data package using `src/05_build_publication_dataset.py`.
6. Run the relevant scripts in `validation/` to reproduce the technical-validation figures.

The exact input files required by each step depend on the source data and intermediate outputs available to the researcher. Restricted toll records, internal station-ID mapping files, weather-fusion intermediate files, and GIS shapefiles must be obtained separately through the appropriate sources and permissions; they are not distributed in this repository.

## Reproducibility notes

- The public traffic–weather dataset uses anonymized station identifiers and reduced-precision station coordinates.
- The original vehicle-level toll transaction records and internal station mappings are not included in this repository.
- The spatial-distribution script requires local GIS boundary and road-network shapefiles.
- The validation scripts operate on the relevant processed station-hour data; use the column names and units documented in the associated dataset.
- Running the complete processing workflow requires access to the relevant input and intermediate files. Public availability of this code does not imply public availability of restricted source data.

## License

The source code in this repository is distributed under the MIT License. See the `LICENSE` file for the full license text. The license for the separately distributed traffic–weather dataset is specified in that dataset's repository record.

## Citation

If you use these scripts, please cite the archived software record:

Feng, M. Source code for: A winter traffic–weather dataset for expressways in cold regions. Figshare (2026). https://doi.org/10.6084/m9.figshare.35014045.

Please also cite the processed traffic–weather dataset separately: Feng, M. ColdRegion_Expressway_TrafficWeather_Dataset. *figshare* https://doi.org/10.6084/m9.figshare.32215491 (2026).

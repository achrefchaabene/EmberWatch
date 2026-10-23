# Tunisia real-data pipeline

This pipeline builds a georeferenced next-day wildfire dataset for Tunisia. It
does not reuse the Kaggle TFRecord coordinates because those files contain no
geographic metadata.

## Data contract

| Role | Source | Why it is needed |
|---|---|---|
| Fire labels | NASA FIRMS, VIIRS 375 m, science-quality archive | Defines observed active-fire locations and the next-day target. |
| Meteorology | ERA5 / ERA5-Land | Supplies temperature, rainfall, humidity and wind on each date. |
| Terrain | Copernicus DEM GLO-30 | Supplies static elevation and slope. |
| Land cover / vegetation | ESA WorldCover and Sentinel-2 NDVI | Restricts analysis to burnable land and represents vegetation. |

Use 2019–2023 for training and 2024–2025 as a held-out test period. Rasterise
all sources to the same `EPSG:4326`, 0.01-degree grid; every exported forecast
will then include a bounding box, CRS and pixel resolution.

## Step 1 — get the real fire labels

1. Create a free NASA Earthdata account.
2. In the [NASA FIRMS Archive Download](https://firms.modaps.eosdis.nasa.gov/download/), request the **VIIRS 375 m / S-NPP** CSV for Tunisia, 2019-01-01 through 2025-12-31. Select science-quality data, not real-time data.
3. When NASA emails the download link, save the CSV as:

```text
data\\raw\\tunisia\\firms_viirs_tunisia_2019_2025.csv
```

4. From the project root, run:

```cmd
.venv\\Scripts\\python.exe scripts\\prepare_tunisia_firms_labels.py --input data\\raw\\tunisia\\firms_viirs_tunisia_2019_2025.csv --min-confidence nominal
```

The script filters detections to Tunisia, validates their date and confidence,
and writes a cleaned CSV plus a GeoJSON layer for inspection.

## Step 2 — quality-check the labels

```cmd
.venv\\Scripts\\python.exe scripts\\summarize_tunisia_firms.py
```

This reports the train/test counts and creates
`artifacts/tunisia_firms_temporal_qc.png`. Keep the split chronological: use
2019–2023 for training and reserve 2024–2025 for the final evaluation.

## Next stages

After the label CSV exists, add ERA5 weather, DEM, and vegetation layers; align
them to the target grid; construct next-day labels; then train and evaluate the
same U-Net with a geographic held-out period.

## Step 3 — ERA5 weather

Create a free [Copernicus Climate Data Store](https://cds.climate.copernicus.eu/)
account. Open the [ERA5 hourly single-levels dataset](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels?tab=download), accept its licence, then go to the [CDS API setup page](https://cds.climate.copernicus.eu/en/how-to-api). Copy the displayed two-line configuration to:

```text
C:\\Users\\moham\\.cdsapirc
```

Never paste the personal access token into a notebook, source file, or Git.

Install the download client once:

```cmd
.venv\\Scripts\\pip.exe install cdsapi
```

Download one month at a time to stay below the CDS request-size limit. Start
with January 2019:

```cmd
.venv\\Scripts\\python.exe scripts\\download_era5_tunisia.py --year 2019 --month 1
```

The request downloads five hourly variables over Tunisia: 2 m temperature, 2 m
dewpoint temperature, 10 m east/west wind, 10 m north/south wind, and total
precipitation. It saves one NetCDF file per month in `data/raw/tunisia/era5/`.
CDS packages this request as an archive containing two NetCDF files: `instant`
for temperature/dewpoint/wind and `accum` for precipitation. The downloader
extracts both automatically.

For manual requests, ERA5 requires 0.25-degree grid-aligned boundaries. Use
North `37.75`, West `7.50`, South `30.00`, East `11.75`.

## Step 4 — create daily weather features

When all twelve months of a year are present, aggregate the hourly files:

```cmd
.venv\\Scripts\\python.exe scripts\\aggregate_era5_daily.py --year 2019
```

This creates `data/processed/tunisia/era5_daily/era5_tunisia_daily_2019.nc`.
Its daily layers are mean/min/max temperature, relative humidity, mean/max wind
speed, wind components, and daily rainfall. The geographic grid and CRS are
preserved for the final forecast layer.

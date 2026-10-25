"""Aggregate ESA WorldCover 2021 land-cover classes to the native ERA5 grid.

The script reads six public Cloud Optimized GeoTIFFs remotely in row blocks.
It does not save their large 10-m source files locally; the output is a small
32x18 NetCDF layer with class fractions per ERA5 cell.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import rasterio
import xarray as xr
from rasterio.windows import Window


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_ERA5 = PROJECT_ROOT / "data" / "processed" / "tunisia" / "era5_daily" / "era5_tunisia_daily_2019.nc"
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "processed" / "tunisia" / "worldcover_2021_era5_025.nc"
URL_TEMPLATE = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"
# 3 by 3 degree WorldCover tiles intersecting the buffered Tunisia ERA5 grid.
TILES = ("N30E006", "N30E009", "N33E006", "N33E009", "N36E006", "N36E009")
CLASS_CODES = {
    "tree_cover_fraction": 10,
    "shrubland_fraction": 20,
    "grassland_fraction": 30,
    "cropland_fraction": 40,
    "built_up_fraction": 50,
    "bare_sparse_fraction": 60,
    "water_fraction": 80,
    "wetland_fraction": 90,
}


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate remote ESA WorldCover 2021 data to the Tunisia ERA5 grid.")
    parser.add_argument("--reference-era5", type=Path, default=REFERENCE_ERA5)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rows-per-block", type=int, default=512)
    return parser.parse_args()


def nearest_indices(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Map coordinates to nearest regularly spaced ERA5 coordinate index."""
    step = abs(float(np.diff(reference).mean()))
    if reference[0] > reference[-1]:
        result = np.rint((reference[0] - values) / step).astype(np.int32)
    else:
        result = np.rint((values - reference[0]) / step).astype(np.int32)
    return result


def main() -> None:
    args = parse_arguments()
    if not args.reference_era5.exists():
        raise FileNotFoundError(f"Reference ERA5 file not found: {args.reference_era5}")
    reference = xr.open_dataset(args.reference_era5)
    latitudes = reference.latitude.values.astype(np.float64)
    longitudes = reference.longitude.values.astype(np.float64)
    height, width = len(latitudes), len(longitudes)
    total = np.zeros((height, width), dtype=np.int64)
    class_counts = {name: np.zeros((height, width), dtype=np.int64) for name in CLASS_CODES}

    env_options = {
        "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
        "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
        "GDAL_HTTP_MULTIRANGE": "YES",
        "GDAL_HTTP_MERGE_CONSECUTIVE_RANGES": "YES",
    }
    with rasterio.Env(**env_options):
        for tile in TILES:
            url = URL_TEMPLATE.format(tile=tile)
            print(f"Reading remote WorldCover tile: {tile}")
            with rasterio.open(url) as source:
                columns = np.arange(source.width)
                longitude_values = source.transform.c + (columns + 0.5) * source.transform.a
                lon_indices = nearest_indices(longitude_values, longitudes)
                valid_lon = (lon_indices >= 0) & (lon_indices < width)
                for row_start in range(0, source.height, args.rows_per_block):
                    row_count = min(args.rows_per_block, source.height - row_start)
                    values = source.read(1, window=Window(0, row_start, source.width, row_count))
                    rows = row_start + np.arange(row_count)
                    latitude_values = source.transform.f + (rows + 0.5) * source.transform.e
                    lat_indices = nearest_indices(latitude_values, latitudes)
                    valid_lat = (lat_indices >= 0) & (lat_indices < height)
                    if not valid_lat.any() or not valid_lon.any():
                        continue
                    selected = values[np.ix_(valid_lat, valid_lon)]
                    lat_grid = np.repeat(lat_indices[valid_lat], valid_lon.sum())
                    lon_grid = np.tile(lon_indices[valid_lon], valid_lat.sum())
                    flattened = selected.reshape(-1)
                    valid = flattened != 0
                    np.add.at(total, (lat_grid[valid], lon_grid[valid]), 1)
                    for name, code in CLASS_CODES.items():
                        matches = valid & (flattened == code)
                        np.add.at(class_counts[name], (lat_grid[matches], lon_grid[matches]), 1)

    safe_total = np.maximum(total, 1)
    layers = {name: (("latitude", "longitude"), (counts / safe_total).astype(np.float32)) for name, counts in class_counts.items()}
    layers["burnable_land_fraction"] = (
        ("latitude", "longitude"),
        (class_counts["tree_cover_fraction"] + class_counts["shrubland_fraction"] + class_counts["grassland_fraction"] + class_counts["cropland_fraction"] + class_counts["wetland_fraction"]) / safe_total,
    )
    layers["valid_landcover_fraction"] = (("latitude", "longitude"), (total > 0).astype(np.float32))
    worldcover = xr.Dataset(
        layers,
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "title": "ESA WorldCover 2021 class fractions aligned to Tunisia ERA5 grid",
            "source": "ESA WorldCover 2021 v200, 10 m map",
            "crs": "EPSG:4326",
            "aggregation": "Fraction of valid 10-m WorldCover pixels assigned to each nearest 0.25-degree ERA5 cell",
            "caveat": "This product covers the full buffered ERA5 rectangle. A Tunisia administrative land mask remains a later refinement.",
        },
    )
    for name in worldcover.data_vars:
        worldcover[name].attrs["units"] = "fraction"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    worldcover.to_netcdf(args.output, encoding={name: {"zlib": True, "complevel": 4} for name in worldcover.data_vars})

    print(f"Grid: {height} lat x {width} lon")
    print(f"Mean burnable-land fraction: {float(worldcover.burnable_land_fraction.mean()):.3f}")
    print(f"Mean water fraction: {float(worldcover.water_fraction.mean()):.3f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

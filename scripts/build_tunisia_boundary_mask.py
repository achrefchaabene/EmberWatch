"""Build a fractional Tunisia country mask on the native 0.25-degree ERA5 grid.

Natural Earth supplies an administrative country boundary.  Each coarse ERA5
cell is oversampled into small pixels so the output records the fraction of
the cell that lies inside Tunisia, rather than using an inaccurate centre-only
test around the border.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import requests
import xarray as xr
from rasterio.features import geometry_mask
from rasterio.transform import from_origin


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import ERA5_GRID_RESOLUTION_DEGREES


GRID_REFERENCE = PROJECT_ROOT / "data" / "processed" / "tunisia" / "era5_daily" / "era5_tunisia_daily_2019.nc"
BOUNDARY_PATH = PROJECT_ROOT / "data" / "raw" / "tunisia" / "boundaries" / "naturalearth_admin0_countries_10m.geojson"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "tunisia" / "tunisia_boundary_fraction_era5_025.nc"
NATURAL_EARTH_URL = "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_0_countries.geojson"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-reference", type=Path, default=GRID_REFERENCE)
    parser.add_argument("--boundary", type=Path, default=BOUNDARY_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--subcells", type=int, default=20, help="Samples per ERA5 cell side; default gives 400 samples/cell.")
    return parser.parse_args()


def load_tunisia_geometry(path: Path) -> dict:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(NATURAL_EARTH_URL, timeout=60)
        response.raise_for_status()
        path.write_bytes(response.content)
        print(f"Downloaded boundary: {path}")
    collection = json.loads(path.read_text(encoding="utf-8"))
    for feature in collection["features"]:
        properties = feature.get("properties", {})
        if properties.get("ADM0_A3") == "TUN" or properties.get("ISO_A3") == "TUN":
            return feature["geometry"]
    raise ValueError("Tunisia (TUN) was not found in the Natural Earth country boundary file.")


def fraction_in_tunisia(latitude: float, longitude: float, geometry: dict, subcells: int) -> float:
    resolution = ERA5_GRID_RESOLUTION_DEGREES
    west = longitude - resolution / 2
    north = latitude + resolution / 2
    transform = from_origin(west, north, resolution / subcells, resolution / subcells)
    inside = geometry_mask([geometry], out_shape=(subcells, subcells), transform=transform, invert=True)
    return float(inside.mean())


def main() -> None:
    args = parse_arguments()
    if args.subcells < 2:
        raise ValueError("--subcells must be at least 2.")
    if not args.grid_reference.exists():
        raise FileNotFoundError(f"ERA5 reference grid not found: {args.grid_reference}")

    geometry = load_tunisia_geometry(args.boundary)
    grid = xr.open_dataset(args.grid_reference)
    latitudes = grid.latitude.values.astype(float)
    longitudes = grid.longitude.values.astype(float)
    fraction = np.empty((len(latitudes), len(longitudes)), dtype=np.float32)
    for row, latitude in enumerate(latitudes):
        for column, longitude in enumerate(longitudes):
            fraction[row, column] = fraction_in_tunisia(latitude, longitude, geometry, args.subcells)

    dataset = xr.Dataset(
        {
            "tunisia_fraction": (("latitude", "longitude"), fraction),
            "is_tunisia_cell": (("latitude", "longitude"), (fraction >= 0.5).astype(np.int8)),
        },
        coords={"latitude": latitudes, "longitude": longitudes},
        attrs={
            "title": "Fractional Tunisia mask on the native ERA5 0.25-degree grid",
            "crs": "EPSG:4326",
            "boundary_source": "Natural Earth 1:10m Admin 0 Countries",
            "country_code": "TUN",
            "method": f"{args.subcells} x {args.subcells} raster samples per ERA5 cell",
        },
    )
    dataset["tunisia_fraction"].attrs["description"] = "Fraction of each ERA5 cell inside Tunisia according to the Natural Earth boundary."
    dataset["is_tunisia_cell"].attrs["description"] = "1 when at least 50 percent of the ERA5 cell lies inside Tunisia."
    args.output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(args.output, encoding={"tunisia_fraction": {"zlib": True, "complevel": 4}})
    print(f"Grid: {len(latitudes)} lat x {len(longitudes)} lon")
    print(f"Tunisia-dominant cells: {int((fraction >= 0.5).sum())}")
    print(f"Total Tunisia-equivalent cells: {float(fraction.sum()):.2f}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()

"""Aggregate monthly ERA5 files into daily, georeferenced Tunisia features."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import xarray as xr


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ERA5_DIR = PROJECT_ROOT / "data" / "raw" / "tunisia" / "era5"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "tunisia" / "era5_daily"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate monthly ERA5 files to daily features.")
    parser.add_argument("--year", type=int, required=True, choices=range(2019, 2026))
    return parser.parse_args()


def open_months(paths: list[Path]) -> xr.Dataset:
    if len(paths) != 12:
        raise FileNotFoundError(f"Expected 12 monthly files, found {len(paths)}: {paths}")
    datasets = [xr.open_dataset(path) for path in paths]
    return xr.concat(datasets, dim="valid_time").sortby("valid_time")


def relative_humidity(temperature_kelvin: xr.DataArray, dewpoint_kelvin: xr.DataArray) -> xr.DataArray:
    """Approximate 2 m relative humidity (%) from temperature and dew point."""
    temperature_celsius = temperature_kelvin - 273.15
    dewpoint_celsius = dewpoint_kelvin - 273.15
    exponent = (17.625 * dewpoint_celsius / (243.04 + dewpoint_celsius)) - (
        17.625 * temperature_celsius / (243.04 + temperature_celsius)
    )
    return (100 * np.exp(exponent)).clip(min=0, max=100)


def main() -> None:
    args = parse_arguments()
    instantaneous = sorted(ERA5_DIR.glob(f"era5_tunisia_{args.year}_??_instant.nc"))
    accumulated = sorted(ERA5_DIR.glob(f"era5_tunisia_{args.year}_??_accum.nc"))
    output_path = OUTPUT_DIR / f"era5_tunisia_daily_{args.year}.nc"
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    instant_data = open_months(instantaneous)
    precipitation_data = open_months(accumulated)

    temperature_celsius = instant_data["t2m"] - 273.15
    wind_speed = np.hypot(instant_data["u10"], instant_data["v10"])
    humidity = relative_humidity(instant_data["t2m"], instant_data["d2m"])

    daily = xr.Dataset(
        {
            "temperature_mean_c": temperature_celsius.resample(valid_time="1D").mean(),
            "temperature_max_c": temperature_celsius.resample(valid_time="1D").max(),
            "temperature_min_c": temperature_celsius.resample(valid_time="1D").min(),
            "relative_humidity_mean_pct": humidity.resample(valid_time="1D").mean(),
            "wind_u_mean_mps": instant_data["u10"].resample(valid_time="1D").mean(),
            "wind_v_mean_mps": instant_data["v10"].resample(valid_time="1D").mean(),
            "wind_speed_mean_mps": wind_speed.resample(valid_time="1D").mean(),
            "wind_speed_max_mps": wind_speed.resample(valid_time="1D").max(),
            # ERA5 total precipitation is metres of water equivalent. Each time
            # step here is hourly accumulation, so sum then convert m to mm.
            "precipitation_sum_mm": 1000 * precipitation_data["tp"].resample(valid_time="1D").sum(),
        }
    )
    daily.attrs.update(
        title="Daily ERA5 weather features for Tunisia",
        source="Copernicus Climate Data Store: reanalysis-era5-single-levels",
        crs="EPSG:4326",
        spatial_resolution="0.25 degrees",
        year=str(args.year),
    )
    daily.to_netcdf(output_path)

    print(f"Input hours: {instant_data.sizes['valid_time']:,}")
    print(f"Output days: {daily.sizes['valid_time']:,}")
    print(f"Grid: {daily.sizes['latitude']} lat x {daily.sizes['longitude']} lon")
    print("Variables:", ", ".join(daily.data_vars))
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()

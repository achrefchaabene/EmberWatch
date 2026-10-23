"""Download one month of ERA5 hourly weather data for Tunisia from CDS.

Before running, create a CDS account, accept the ERA5 licence in the browser,
and save the personal access token in C:\\Users\\<user>\\.cdsapirc.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.tunisia import ERA5_TUNISIA_BBOX


OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "tunisia" / "era5"
ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "total_precipitation",
]


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download one ERA5 month for Tunisia.")
    parser.add_argument("--year", type=int, required=True, choices=range(2019, 2026))
    parser.add_argument("--month", type=int, required=True, choices=range(1, 13))
    return parser.parse_args()


def main() -> None:
    args = parse_arguments()
    try:
        import cdsapi
    except ImportError as error:
        raise SystemExit(
            "Missing cdsapi. Install it with: .venv\\Scripts\\pip.exe install cdsapi"
        ) from error

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    stem = f"era5_tunisia_{args.year}_{args.month:02d}"
    archive_path = OUTPUT_DIR / f"{stem}.zip"
    # Version 1 of this script used the .nc extension for a CDS ZIP response.
    # Reuse it rather than downloading January 2019 again.
    legacy_archive_path = OUTPUT_DIR / f"{stem}.nc"
    if not archive_path.exists() and legacy_archive_path.exists() and zipfile.is_zipfile(legacy_archive_path):
        archive_path = legacy_archive_path

    request = {
        "product_type": ["reanalysis"],
        "variable": ERA5_VARIABLES,
        "year": [str(args.year)],
        "month": [f"{args.month:02d}"],
        "day": [f"{day:02d}" for day in range(1, 32)],
        "time": [f"{hour:02d}:00" for hour in range(24)],
        # CDS expects North, West, South, East in WGS84.
        "area": [
            ERA5_TUNISIA_BBOX["north"],
            ERA5_TUNISIA_BBOX["west"],
            ERA5_TUNISIA_BBOX["south"],
            ERA5_TUNISIA_BBOX["east"],
        ],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    if not archive_path.exists():
        print(f"Requesting ERA5 for Tunisia, {args.year}-{args.month:02d}. This may take several minutes.")
        client = cdsapi.Client()
        client.retrieve("reanalysis-era5-single-levels", request).download(str(archive_path))
        print(f"Saved archive: {archive_path}")
    else:
        print(f"Using existing CDS archive: {archive_path}")

    if not zipfile.is_zipfile(archive_path):
        raise ValueError(f"Expected a ZIP archive from CDS, got: {archive_path}")
    extracted_paths = []
    with zipfile.ZipFile(archive_path) as archive:
        for member in archive.infolist():
            if not member.filename.endswith(".nc"):
                continue
            suffix = "instant" if "stepType-instant" in member.filename else "accum"
            output_path = OUTPUT_DIR / f"{stem}_{suffix}.nc"
            with archive.open(member) as source, output_path.open("wb") as output:
                shutil.copyfileobj(source, output)
            extracted_paths.append(output_path)

    if not extracted_paths:
        raise ValueError(f"No NetCDF files found inside: {archive_path}")
    for output_path in extracted_paths:
        print(f"Saved NetCDF: {output_path}")


if __name__ == "__main__":
    main()

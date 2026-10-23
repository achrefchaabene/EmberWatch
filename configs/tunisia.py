"""Spatial and temporal settings for the Tunisia wildfire pipeline."""

# Country-level guardrail in WGS84 (longitude/latitude) for FIRMS points.
TUNISIA_BBOX = {
    "west": 7.52,
    "south": 30.23,
    "east": 11.60,
    "north": 37.54,
}

# ERA5 requests must align to its 0.25-degree grid.  This buffered rectangle
# fully contains Tunisia and is intentionally wider than the FIRMS guardrail.
ERA5_TUNISIA_BBOX = {
    "west": 7.50,
    "south": 30.00,
    "east": 11.75,
    "north": 37.75,
}

# VIIRS 375 m data are available since 2012.  Starting at 2019 gives several
# years for training while leaving newer data for evaluation.
TRAIN_START = "2019-01-01"
TRAIN_END = "2023-12-31"
TEST_START = "2024-01-01"
TEST_END = "2025-12-31"

# Target grid used when all source layers are rasterised and aligned.
TARGET_CRS = "EPSG:4326"
TARGET_RESOLUTION_DEGREES = 0.01  # approximately 1 km in Tunisia

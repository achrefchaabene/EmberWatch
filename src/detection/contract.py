"""Data contract for detections and alerts (Cahier des charges §8 / §8.1).

Conventions (frozen in contract v1):
- time: UTC ISO-8601 for camera frames
- boxes: [x1, y1, x2, y2] in ORIGINAL frame pixels
- ids: camera_id, sequence_id, alert_id (UUID)
- every record carries schema_version and the model version
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, Field, field_validator

SCHEMA_VERSION = "1.0"

# Alert lifecycle: an alert is NEVER acted on automatically (Responsible-AI principle P1).
AlertStatus = Literal["PENDING_CONFIRMATION", "CONFIRMED", "DISMISSED"]


class Detection(BaseModel):
    """One smoke box found in one frame."""

    box: list[float] = Field(..., description="[x1, y1, x2, y2] in original frame pixels")
    confidence: float = Field(..., ge=0.0, le=1.0)
    label: str = "smoke"

    @field_validator("box")
    @classmethod
    def _check_box(cls, box: list[float]) -> list[float]:
        # A box must have 4 coordinates with x2 > x1 and y2 > y1.
        if len(box) != 4:
            raise ValueError("box must be [x1, y1, x2, y2]")
        x1, y1, x2, y2 = box
        if x2 <= x1 or y2 <= y1:
            raise ValueError("box must satisfy x2 > x1 and y2 > y1")
        return [round(float(v), 1) for v in box]


class GeoLocation(BaseModel):
    """Filled by M5 (Achref). Kept optional so M2 can emit alerts before M5 exists."""

    lat: float
    lon: float
    uncertainty_m: float
    method: str = "DEM_RAY"


class Alert(BaseModel):
    """Alert record exactly as in CDC §8.1 (M3 + M5)."""

    schema_version: str = SCHEMA_VERSION
    alert_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    camera_id: str
    sequence_id: str | None = None
    t_first_detection: str  # UTC ISO-8601, e.g. 2024-07-18T16:24:10Z
    confidence: float = Field(..., ge=0.0, le=1.0)
    box: list[float]
    tiles: list[int] = Field(default_factory=list)
    evidence_frames: list[str] = Field(default_factory=list)
    bearing_deg: float | None = None  # M5
    elevation_deg: float | None = None  # M5
    location: GeoLocation | None = None  # M5
    status: AlertStatus = "PENDING_CONFIRMATION"
    models: dict[str, str] = Field(default_factory=dict)


def utc_now_iso() -> str:
    """Current UTC time in the contract format (seconds precision, trailing Z)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def box_to_tiles(box: list[float], image_w: int, image_h: int, tile_size: int = 224) -> list[int]:
    """Return the ids of the fixed-grid tiles a box overlaps.

    Tiles are numbered row by row on a non-overlapping grid of `tile_size` pixels
    (SmokeyNet-style). The same grid will be used by the M3 tile-level model, so an
    alert can always point to the tiles that triggered it (CDC M3 requirement).
    """
    cols = max(1, -(-image_w // tile_size))  # ceil division
    x1, y1, x2, y2 = box
    # Clamp to the image, then convert pixel range -> tile index range.
    c1 = int(max(0, x1) // tile_size)
    c2 = int(min(image_w - 1, x2) // tile_size)
    r1 = int(max(0, y1) // tile_size)
    r2 = int(min(image_h - 1, y2) // tile_size)
    return [r * cols + c for r in range(r1, r2 + 1) for c in range(c1, c2 + 1)]


def build_alert(
    detections: list[Detection],
    camera_id: str,
    image_w: int,
    image_h: int,
    model_version: str,
    frame_name: str | None = None,
    timestamp: str | None = None,
    sequence_id: str | None = None,
) -> Alert | None:
    """Turn the detections of one frame into an alert (strongest box), or None.

    Single-frame alert = M2 baseline. The M3 persistence rule (k of the last n
    frames) will decide when to call this in Sprint 2.
    """
    if not detections:
        return None
    best = max(detections, key=lambda d: d.confidence)
    return Alert(
        camera_id=camera_id,
        sequence_id=sequence_id,
        t_first_detection=timestamp or utc_now_iso(),
        confidence=round(best.confidence, 4),
        box=best.box,
        tiles=box_to_tiles(best.box, image_w, image_h),
        evidence_frames=[frame_name] if frame_name else [],
        models={"M2": model_version},
    )

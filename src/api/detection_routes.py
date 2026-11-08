"""M2 detection endpoints (owner: Mouhib).

POST /api/v1/detect   -> run the smoke detector on one uploaded frame
GET  /api/v1/alerts   -> latest alerts created by /detect (in memory until M7 adds PostGIS)

The detector is loaded once, lazily, from EMBERWATCH_DETECTOR_WEIGHTS
(default: artifacts/m2_smoke_best.pt). Weights are not versioned in git.
"""

from __future__ import annotations

import io
import os
from collections import deque
from functools import lru_cache
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from src.detection.contract import Alert, Detection, build_alert

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS = PROJECT_ROOT / "artifacts" / "m2_smoke_best.pt"
MAX_ALERTS = 200  # in-memory history size

router = APIRouter(prefix="/api/v1", tags=["detection"])
_alerts: deque[Alert] = deque(maxlen=MAX_ALERTS)


@lru_cache(maxsize=1)
def get_detector():
    """Load the detector once per process (overridden in tests)."""
    weights = Path(os.environ.get("EMBERWATCH_DETECTOR_WEIGHTS", DEFAULT_WEIGHTS))
    if not weights.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Detector weights not found at {weights}. Train M2 or set EMBERWATCH_DETECTOR_WEIGHTS.",
        )
    from src.detection.detector import SmokeDetector

    return SmokeDetector(
        weights,
        conf=float(os.environ.get("EMBERWATCH_DETECTOR_CONF", 0.25)),
        device=os.environ.get("EMBERWATCH_DEVICE", "cpu"),
    )


def _decode_image(raw: bytes) -> np.ndarray:
    """Decode uploaded bytes to an RGB array, or reject the request."""
    from PIL import Image

    try:
        return np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
    # Broad on purpose: once Ultralytics is imported it patches Image.open, and an
    # unreadable file can raise ModuleNotFoundError (pi_heif) instead of PIL's
    # UnidentifiedImageError. Any decode failure is a client error, not a 500.
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail="File is not a readable image.") from exc


@router.post("/detect")
async def detect(
    image: UploadFile = File(..., description="One camera frame (jpg/png)"),
    camera_id: str = Form("CAM-UNKNOWN"),
    timestamp: str | None = Form(None, description="Frame time, UTC ISO-8601"),
    detector=Depends(get_detector),
) -> dict:
    """Detect smoke on one frame. Returns all boxes + an alert if smoke was found.

    Single-frame alerts are the M2 baseline; M3 will add the k-of-n persistence rule.
    The alert is always PENDING_CONFIRMATION (human confirmation, principle P1).
    """
    frame = _decode_image(await image.read())
    h, w = frame.shape[:2]
    detections: list[Detection] = detector.predict(frame)
    alert = build_alert(
        detections,
        camera_id=camera_id,
        image_w=w,
        image_h=h,
        model_version=detector.model_version,
        frame_name=image.filename,
        timestamp=timestamp,
    )
    if alert:
        _alerts.append(alert)
    return {
        "camera_id": camera_id,
        "image_size": [w, h],
        "model": detector.model_version,
        "detections": [d.model_dump() for d in detections],
        "alert": alert.model_dump() if alert else None,
    }


@router.get("/alerts")
def list_alerts(limit: int = 20) -> list[dict]:
    """Newest alerts first."""
    return [a.model_dump() for a in list(_alerts)[::-1][: max(1, min(limit, MAX_ALERTS))]]

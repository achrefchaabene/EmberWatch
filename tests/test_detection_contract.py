"""Unit tests for the M2 data contract (CDC §8.1)."""

import pytest
from pydantic import ValidationError

from src.detection.contract import Detection, box_to_tiles, build_alert


def test_box_must_be_ordered():
    # x2 <= x1 is not a valid box
    with pytest.raises(ValidationError):
        Detection(box=[100, 10, 50, 40], confidence=0.9)


def test_tiles_cover_box_on_224_grid():
    # 1000 px wide -> 5 tile columns; box spans cols 1-2 on row 0
    assert box_to_tiles([300, 10, 500, 100], image_w=1000, image_h=600) == [1, 2]
    # box crossing a row boundary spans rows 0-1 -> ids 1,2 and 6,7
    assert box_to_tiles([300, 200, 500, 250], image_w=1000, image_h=600) == [1, 2, 6, 7]


def test_no_detection_no_alert():
    assert build_alert([], "CAM-03", 1000, 600, "yolo-smoke@test") is None


def test_alert_uses_strongest_box_and_contract_fields():
    dets = [Detection(box=[0, 0, 10, 10], confidence=0.4), Detection(box=[300, 10, 500, 100], confidence=0.92)]
    alert = build_alert(dets, "CAM-03", 1000, 600, "yolo-smoke@test", frame_name="t0.jpg",
                        timestamp="2024-07-18T16:24:10Z")
    data = alert.model_dump()
    assert data["schema_version"] == "1.0"
    assert data["status"] == "PENDING_CONFIRMATION"  # never auto-confirmed (P1)
    assert data["box"] == [300.0, 10.0, 500.0, 100.0]
    assert data["confidence"] == 0.92
    assert data["tiles"] == [1, 2]
    assert data["models"] == {"M2": "yolo-smoke@test"}
    assert data["evidence_frames"] == ["t0.jpg"]
    assert data["location"] is None  # filled later by M5

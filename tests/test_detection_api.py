"""API tests for /api/v1/detect with a fake detector (no weights, no GPU needed)."""

import io

import pytest

from fastapi.testclient import TestClient
from PIL import Image

from src.api.detection_routes import get_detector
from src.api.main import app
from src.detection.contract import Detection


class FakeDetector:
    """Returns one fixed smoke box, or nothing."""

    model_version = "yolo-smoke@fake"

    def __init__(self, found: bool) -> None:
        self.found = found

    def predict(self, frame):
        return [Detection(box=[300, 10, 500, 100], confidence=0.9)] if self.found else []


def _png(w=1000, h=600) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (120, 140, 160)).save(buf, format="PNG")
    return buf.getvalue()


def _client(found: bool) -> TestClient:
    app.dependency_overrides[get_detector] = lambda: FakeDetector(found)
    return TestClient(app)


def test_detect_creates_pending_alert():
    client = _client(found=True)
    r = client.post("/api/v1/detect", files={"image": ("t0.png", _png(), "image/png")}, data={"camera_id": "CAM-03"})
    assert r.status_code == 200
    body = r.json()
    assert body["image_size"] == [1000, 600]
    assert body["alert"]["status"] == "PENDING_CONFIRMATION"
    assert body["alert"]["camera_id"] == "CAM-03"
    assert body["alert"]["tiles"] == [1, 2]
    assert client.get("/api/v1/alerts").json()[0]["alert_id"] == body["alert"]["alert_id"]


def test_detect_without_smoke_returns_no_alert():
    r = _client(found=False).post("/api/v1/detect", files={"image": ("t.png", _png(), "image/png")})
    assert r.status_code == 200 and r.json()["alert"] is None


def test_detect_rejects_non_image():
    r = _client(found=True).post("/api/v1/detect", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_detect_rejects_non_image_after_ultralytics_import():
    # Regression: Ultralytics patches PIL.Image.open; a bad file must still give 400, not 500.
    pytest.importorskip("ultralytics")
    import ultralytics  # noqa: F401  (import applies the patch)

    r = _client(found=True).post("/api/v1/detect", files={"image": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 400


def test_forecast_routes_still_work():
    # M6 (Achref) endpoints must be unaffected by the M2 router.
    assert _client(found=True).get("/health").json() == {"status": "ok"}

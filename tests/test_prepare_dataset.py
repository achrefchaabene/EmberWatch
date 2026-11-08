"""Tests for the M2 dataset builder on a tiny synthetic D-Fire + HPWREN layout."""

import json
from pathlib import Path

from src.detection.prepare_dataset import build, event_key


def _write(root: Path, split: str, name: str, rows: list[str]) -> None:
    """Create one fake image + YOLO label file."""
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    (root / split / "images" / f"{name}.jpg").write_bytes(b"fake")
    (root / split / "labels" / f"{name}.txt").write_text("\n".join(rows))


def test_event_key_groups_frames_of_same_fire():
    a = event_key("20190529_94Fire_lp-s-mobo-c__2019-05-29T16-08-40_jpg.rf.0a1b2c")
    b = event_key("20190529_94Fire_lp-s-mobo-c__2019-05-29T16-09-40")
    assert a == b == "20190529_94Fire_lp-s-mobo-c"
    assert event_key("AoF06723") == "AoF06723"


def test_build_keeps_smoke_only_and_is_leakage_free(tmp_path):
    dfire, hpwren, out = tmp_path / "dfire", tmp_path / "hpwren", tmp_path / "out"
    # D-Fire: class 0 appears in more images than class 1 -> 0 is smoke.
    for i in range(12):
        _write(dfire, "train", f"img{i}", ["0 0.5 0.5 0.1 0.1"] + (["1 0.2 0.2 0.1 0.1"] if i < 3 else []))
    _write(dfire, "train", "fire_only", ["1 0.3 0.3 0.1 0.1"])
    _write(dfire, "test", "t0", ["0 0.5 0.5 0.2 0.2"])
    # HPWREN: 5 events x 4 frames, Roboflow-style data.yaml.
    for e in range(5):
        for f in range(4):
            _write(hpwren, "train", f"2019_fire{e}_cam__2019-06-0{f}T10-00-00", ["0 0.4 0.4 0.05 0.05"])
    (hpwren / "data.yaml").write_text("nc: 1\nnames: ['smoke']\n")

    report = build(dfire, hpwren, out, val=0.2, test=0.2, seed=0, copy=True)

    assert report["sources"]["dfire"]["smoke_class_id"] == 0
    # no image is lost: every input image lands in exactly one split
    total_in = report["sources"]["dfire"]["images"] + report["sources"]["hpwren"]["images"]
    assert sum(s["images"] for s in report["splits"].values()) == total_in
    assert report["leakage"]["leakage_free"] is True
    # fire-only image is kept as a negative with an empty label file
    neg = next((out / "labels").rglob("dfire__fire_only.txt"))
    assert neg.read_text() == ""
    # every written label is class 0 (smoke) only
    for label in (out / "labels").rglob("*.txt"):
        assert all(line.startswith("0 ") for line in label.read_text().splitlines())
    # official D-Fire test image stays in test
    assert (out / "labels" / "test" / "dfire__t0.txt").exists()
    assert "names:\n  0: smoke" in (out / "data.yaml").read_text()
    assert json.loads((out / "split_report.json").read_text())["leakage"]["leakage_free"]

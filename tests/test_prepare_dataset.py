"""Tests for the M2 dataset builder on a tiny synthetic D-Fire + HPWREN layout."""

import json
import zlib
from pathlib import Path

import numpy as np
from PIL import Image

from src.detection.prepare_dataset import build, event_key


def _write(root: Path, split: str, name: str, rows: list[str]) -> None:
    """Create one fake image + YOLO label file."""
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "labels").mkdir(parents=True, exist_ok=True)
    # Real, distinct images: the builder now drops undecodable files and byte-identical copies.
    rng = np.random.default_rng(zlib.crc32(name.encode()))  # unique, reproducible pixels per name
    img = Image.fromarray(rng.integers(0, 256, (32, 32, 3), dtype=np.uint8))
    img.save(root / split / "images" / f"{name}.png")
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


def test_hpwren_random_names_are_grouped_by_scene(tmp_path):
    """Roboflow names (random ids) carry no fire event: frames of the same camera scene
    must still land in ONE split, and different scenes must not be merged."""
    root, out = tmp_path / "hpwren", tmp_path / "out"
    rng = np.random.default_rng(1)
    (root / "train" / "images").mkdir(parents=True)
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("names: ['smoke']\n")
    scene_of = {}
    for scene in range(5):
        # Each scene = its own random landscape (distinct fingerprints between scenes)
        base = rng.integers(0, 256, (12, 16, 3), dtype=np.uint8).repeat(20, axis=0).repeat(20, axis=1)
        for frame in range(4):
            img = base.copy()
            img[40:60, 100 + 10 * frame:120 + 10 * frame] = 230  # small plume drifting = same scene
            name = "ck" + "".join(rng.choice(list("abcdefghij0123456789"), 20)) + "_jpeg.rf.0f3a"
            Image.fromarray(img).save(root / "train" / "images" / f"{name}.jpg")
            (root / "train" / "labels" / f"{name}.txt").write_text("0 0.5 0.2 0.1 0.1\n")
            scene_of[f"hpwren__{name}"] = scene

    report = build(None, root, out, val=0.2, test=0.2, seed=0, copy=True)

    grouping = report["sources"]["hpwren"]["grouping"]
    assert grouping["method"].startswith("visual similarity")
    assert grouping["groups"] == 5  # 5 scenes found, none merged
    split_of_scene: dict[int, set] = {}
    for split in ("train", "val", "test"):
        for f in (out / "labels" / split).glob("*.txt"):
            split_of_scene.setdefault(scene_of[f.stem], set()).add(split)
    assert all(len(s) == 1 for s in split_of_scene.values())  # no scene crosses splits
    assert report["leakage"]["leakage_free"]

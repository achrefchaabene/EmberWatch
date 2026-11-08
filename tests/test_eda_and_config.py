"""Tests for the dataset study (EDA), dataset cleaning and the training config."""

from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from src.detection.eda import analyse, save_report
from src.detection.prepare_dataset import build, clean_box
from src.detection.train import DEFAULT_CONFIG, build_albumentations, load_config


def _img(path: Path, value: int, gray: bool = False, size=(320, 240)) -> None:
    """Write a simple RGB (or grey) image with a gradient so it is not trivially identical."""
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.zeros((size[1], size[0], 3), np.uint8)
    a[..., 0] = value
    a[..., 1] = np.linspace(0, 255, size[0], dtype=np.uint8)[None, :]
    a[..., 2] = 255 - value
    if gray:
        a[...] = a.mean(axis=2, keepdims=True).astype(np.uint8)
    Image.fromarray(a).save(path)


def _dataset(root: Path) -> None:
    for i in range(6):
        _img(root / "train" / "images" / f"a{i}.jpg", 40 * i, gray=(i == 0))
        (root / "train" / "labels" / f"a{i}.txt").parent.mkdir(parents=True, exist_ok=True)
        (root / "train" / "labels" / f"a{i}.txt").write_text("0 0.5 0.5 0.05 0.05\n" if i else "")
    _img(root / "test" / "images" / "t0.jpg", 225)  # not equal to any train value
    (root / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (root / "test" / "labels" / "t0.txt").write_text("0 0.5 0.5 0.5 0.5\n0 0.5 0.5 0 0.1\n")  # 2nd row invalid
    (root / "train" / "images" / "broken.jpg").write_bytes(b"nope")


def test_eda_measures_and_recommends(tmp_path):
    _dataset(tmp_path / "ds")
    stats = analyse(tmp_path / "ds")
    assert stats["corrupt_images"] == 1
    assert stats["invalid_label_rows"] == 1
    assert stats["grayscale_share"] == pytest.approx(1 / 7, abs=0.01)
    assert stats["boxes"]["small_share"] > 0  # 0.05 * 320 px = 16 px boxes are "small"
    topics = {r["topic"] for r in stats["recommendations"]}
    assert {"cleaning", "small objects", "grayscale", "leakage", "resize"} <= topics
    save_report(stats, tmp_path / "out")
    for f in ("eda.json", "eda.md", "box_sizes.png", "box_centres.png", "image_quality.png"):
        assert (tmp_path / "out" / f).exists()


def test_clean_box_clips_and_drops():
    assert clean_box(["0", "0.98", "0.5", "0.1", "0.1"]) == ["0", "0.965000", "0.500000", "0.070000", "0.100000"]
    assert clean_box(["0", "0.5", "0.5", "0", "0.1"]) is None  # zero width
    assert clean_box(["0", "x", "0.5", "0.1", "0.1"]) is None  # not a number


def test_prepare_removes_cross_split_copies_and_corrupt(tmp_path):
    raw = tmp_path / "raw"
    _dataset(raw)
    # Same bytes under another name in test = leakage the name-based check cannot see.
    (raw / "test" / "images" / "copy.jpg").write_bytes((raw / "train" / "images" / "a3.jpg").read_bytes())
    (raw / "test" / "labels" / "copy.txt").write_text("0 0.5 0.5 0.05 0.05\n")
    report = build(raw, None, tmp_path / "out", val=0.2, test=0.0, seed=0, copy=True)
    removed = report["removed_cross_split_duplicates"]
    assert removed["test"] + removed["val"] >= 1
    assert report["splits"]["train"].get("dropped_corrupt_images") == 1
    assert report["splits"]["test"].get("dropped_invalid_boxes") == 1
    assert report["leakage"]["leakage_free"]


def test_prepare_resizes_only_large_images(tmp_path):
    raw = tmp_path / "raw"
    _img(raw / "train" / "images" / "big.jpg", 10, size=(2000, 1000))
    (raw / "train" / "labels").mkdir(parents=True, exist_ok=True)
    (raw / "train" / "labels" / "big.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    _img(raw / "test" / "images" / "t.jpg", 10)
    (raw / "test" / "labels").mkdir(parents=True, exist_ok=True)
    (raw / "test" / "labels" / "t.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    build(raw, None, tmp_path / "out", val=0.0, test=0.0, seed=0, copy=True, max_side=1280)
    with Image.open(next((tmp_path / "out" / "images" / "train").glob("*big*"))) as im:
        assert max(im.size) == 1280  # resized, labels (normalised) stay valid
    with Image.open(next((tmp_path / "out" / "images" / "test").glob("*"))) as im:
        assert im.size == (320, 240)  # small image untouched


def test_config_overrides_and_safe_augmentation_choices():
    cfg = load_config(DEFAULT_CONFIG, {"epochs": 3, "batch": None})
    assert cfg["epochs"] == 3 and cfg["batch"] == 32  # None keeps the YAML value
    # Domain rules that must never be changed by accident:
    assert cfg["flipud"] == 0.0  # smoke rises
    assert cfg["degrees"] == 0.0  # levelled tower cameras
    assert cfg["mixup"] == 0.0  # semi-transparent ghosts look like smoke
    assert cfg["optimizer"] == "SGD"  # fixed for comparable ablations
    assert "time" not in cfg  # would silently change the number of epochs


def test_every_yaml_key_is_a_real_ultralytics_setting():
    # A typo (e.g. 'hsv_hh') would crash a Kaggle run after the dataset download.
    pytest.importorskip("ultralytics")
    import ultralytics.cfg as ucfg

    defaults = yaml.safe_load((Path(ucfg.__file__).parent / "default.yaml").read_text())
    ours = yaml.safe_load(DEFAULT_CONFIG.read_text())
    unknown = set(ours) - set(defaults) - {"albumentations"}
    assert not unknown, f"Unknown Ultralytics keys: {unknown}"


def test_albumentations_list_skips_disabled_transforms():
    pytest.importorskip("albumentations")
    transforms = build_albumentations({"to_gray": 0.1, "blur": 0.0, "clahe": 0.0})
    assert [type(t).__name__ for t in transforms] == ["ToGray"]

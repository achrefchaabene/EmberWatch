"""Build the M2 smoke-only YOLO dataset from D-Fire + HPWREN smoke boxes.

Why:
- M2 detects SMOKE only (the CDC targets early smoke, flames come later).
- D-Fire images with fire but no smoke are kept as negatives (empty label file).
- HPWREN frames come from a few fire events: they are split BY EVENT so frames of
  the same fire never appear in train and test (CDC M1 "no leakage" rule).
- D-Fire keeps its official test split so our mAP is comparable with the paper.

Expected inputs (YOLO format, as downloaded from Kaggle / Roboflow):
    <root>/<split>/images/*.jpg  and  <root>/<split>/labels/*.txt
    (<split> = train / valid / val / test; any subset is fine)

Usage:
    python -m src.detection.prepare_dataset --dfire data/raw/dfire \
        --hpwren data/raw/hpwren_smoke --out data/processed/m2_smoke
"""

from __future__ import annotations

import argparse
import json
import random
import re
import shutil
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}

# Known D-Fire statistics (README): images containing smoke vs containing fire.
DFIRE_SMOKE_IMAGES = 5867 + 4658  # only smoke + fire & smoke
DFIRE_FIRE_IMAGES = 1164 + 4658  # only fire + fire & smoke


@dataclass
class Sample:
    """One image with its source, original split and event group."""

    image: Path
    label: Path | None
    source: str
    orig_split: str | None
    event: str


# ----------------------------------------------------------------- collection
def collect_samples(root: Path, source: str) -> list[Sample]:
    """Find every (image, label) pair under root, remembering the original split."""
    samples: list[Sample] = []
    for image in sorted(root.rglob("*")):
        if image.suffix.lower() not in IMAGE_EXTS or "images" not in image.parts:
            continue
        # .../<split>/images/name.jpg -> .../<split>/labels/name.txt
        idx = len(image.parts) - 1 - image.parts[::-1].index("images")
        label = Path(*image.parts[:idx], "labels", *image.parts[idx + 1 :]).with_suffix(".txt")
        split_dir = image.parts[idx - 1].lower() if idx > 0 else ""
        samples.append(
            Sample(
                image=image,
                label=label if label.exists() else None,
                source=source,
                orig_split=SPLIT_ALIASES.get(split_dir),
                event=event_key(image.stem),
            )
        )
    return samples


def event_key(stem: str) -> str:
    """Group frames of the same fire event.

    HPWREN / AI-For-Mankind names look like
    '20190529_94Fire_lp-s-mobo-c__2019-05-29T16-08-40' -> event '20190529_94Fire_lp-s-mobo-c'.
    Roboflow appends '.rf.<hash>' and '_jpg' -> stripped first.
    Names without '__' (e.g. D-Fire 'AoF06723') are their own group.
    """
    stem = re.sub(r"\.rf\.[0-9a-f]+$", "", stem)
    stem = re.sub(r"_(jpg|jpeg|png)$", "", stem)
    return stem.split("__")[0]


def read_labels(path: Path | None) -> list[list[str]]:
    """Read a YOLO label file as rows of strings ('cls cx cy w h')."""
    if path is None:
        return []
    rows = [line.split() for line in path.read_text().splitlines() if line.strip()]
    return [r for r in rows if len(r) == 5]


# --------------------------------------------------------------- class lookup
def detect_smoke_class(samples: list[Sample]) -> tuple[int, dict[int, int]]:
    """Return the D-Fire class id for smoke, inferred from per-class image counts.

    The D-Fire README gives counts (smoke in ~10.5k images, fire in ~5.8k) but not the
    numeric ids, so the class present in MORE images is smoke.
    """
    images_per_class: Counter[int] = Counter()
    for s in samples:
        images_per_class.update({int(r[0]) for r in read_labels(s.label)})
    if not images_per_class:
        raise ValueError("No labels found: check the D-Fire path.")
    smoke_id = images_per_class.most_common(1)[0][0]
    return smoke_id, dict(images_per_class)


def smoke_class_from_yaml(root: Path) -> int | None:
    """Read the smoke class id from a Roboflow data.yaml ('names: [smoke]')."""
    for yaml_path in root.rglob("data.yaml"):
        text = yaml_path.read_text()
        match = re.search(r"names:\s*\[(.*?)\]", text, re.S)
        if match:
            names = [n.strip().strip("'\"").lower() for n in match.group(1).split(",")]
        else:  # block style: "names:\n  0: smoke" or "- smoke"
            names = re.findall(r"^\s*(?:-|\d+:)\s*['\"]?([\w-]+)['\"]?\s*$", text.split("names:")[-1], re.M)
            names = [n.lower() for n in names]
        if "smoke" in names:
            return names.index("smoke")
    return None


# -------------------------------------------------------------------- splits
def group_split(samples: list[Sample], val: float, test: float, seed: int) -> dict[str, list[Sample]]:
    """Split by event group so no event crosses splits."""
    groups: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        groups[s.event].append(s)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test, n_val = round(n * test), round(n * val)
    if n >= 3:  # keep at least one event in a requested split (fraction > 0) when possible
        n_test = max(1, n_test) if test > 0 else 0
        n_val = max(1, n_val) if val > 0 else 0
    parts = {"test": keys[:n_test], "val": keys[n_test : n_test + n_val], "train": keys[n_test + n_val :]}
    return {split: [s for k in ks for s in groups[k]] for split, ks in parts.items()}


def split_dfire(samples: list[Sample], val: float, seed: int) -> dict[str, list[Sample]]:
    """Keep the official D-Fire test split; carve val out of train."""
    out: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    train_pool = []
    for s in samples:
        if s.orig_split == "test":
            out["test"].append(s)
        elif s.orig_split == "val":
            out["val"].append(s)
        else:
            train_pool.append(s)
    if not out["val"]:
        sub = group_split(train_pool, val=val, test=0.0, seed=seed)
        out["train"], out["val"] = sub["train"], sub["val"]
    else:
        out["train"] = train_pool
    return out


# ---------------------------------------------------------------------- write
def write_split(split: str, samples: list[Sample], smoke_id: int, out: Path, copy: bool) -> Counter:
    """Write images (symlink or copy) and smoke-only labels; return box/image counts."""
    stats: Counter = Counter()
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for s in samples:
        name = f"{s.source}__{s.image.stem}"
        dst_img = out / "images" / split / f"{name}{s.image.suffix.lower()}"
        if not dst_img.exists():
            if copy:
                shutil.copy2(s.image, dst_img)
            else:
                dst_img.symlink_to(s.image.resolve())
        # Keep only smoke boxes and remap them to class 0.
        smoke_rows = [["0", *r[1:]] for r in read_labels(s.label) if int(r[0]) == smoke_id]
        (out / "labels" / split / f"{name}.txt").write_text("".join(" ".join(r) + "\n" for r in smoke_rows))
        stats["images"] += 1
        stats["boxes"] += len(smoke_rows)
        stats["negatives"] += int(not smoke_rows)
    return stats


def leakage_report(splits: dict[str, list[Sample]]) -> dict:
    """Check that no event / image appears in two splits (CDC acceptance: leakage report)."""
    events = {k: {(s.source, s.event) for s in v} for k, v in splits.items()}
    images = {k: {s.image.resolve() for s in v} for k, v in splits.items()}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    report = {
        f"{a}&{b}": {"shared_events": len(events[a] & events[b]), "shared_images": len(images[a] & images[b])}
        for a, b in pairs
    }
    report["leakage_free"] = all(v["shared_events"] == 0 and v["shared_images"] == 0 for v in report.values())
    return report


def build(dfire: Path | None, hpwren: Path | None, out: Path, val: float, test: float, seed: int, copy: bool) -> dict:
    """Build the merged dataset and return the report written to split_report.json."""
    if not dfire and not hpwren:
        raise ValueError("Give at least one source (--dfire and/or --hpwren).")
    out.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[tuple[Sample, int]]] = {"train": [], "val": [], "test": []}
    report: dict = {"seed": seed, "sources": {}}

    if dfire:
        samples = collect_samples(dfire, "dfire")
        smoke_id, per_class = detect_smoke_class(samples)
        report["sources"]["dfire"] = {
            "images": len(samples),
            "smoke_class_id": smoke_id,
            "images_per_class": per_class,
            "expected_from_readme": {"smoke": DFIRE_SMOKE_IMAGES, "fire": DFIRE_FIRE_IMAGES},
        }
        for k, v in split_dfire(samples, val, seed).items():
            merged[k] += [(s, smoke_id) for s in v]

    if hpwren:
        samples = collect_samples(hpwren, "hpwren")
        smoke_id = smoke_class_from_yaml(hpwren)
        if smoke_id is None:
            smoke_id, _ = detect_smoke_class(samples)
        report["sources"]["hpwren"] = {
            "images": len(samples),
            "events": len({s.event for s in samples}),
            "smoke_class_id": smoke_id,
        }
        for k, v in group_split(samples, val, test, seed).items():
            merged[k] += [(s, smoke_id) for s in v]

    report["splits"] = {}
    for split, pairs in merged.items():
        stats: Counter = Counter()
        for s, sid in pairs:
            stats += write_split(split, [s], sid, out, copy)
        report["splits"][split] = dict(stats)

    report["leakage"] = leakage_report({k: [s for s, _ in v] for k, v in merged.items()})
    # Ultralytics dataset file: one class, paths relative to this folder.
    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: smoke\n"
    )
    (out / "split_report.json").write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dfire", type=Path, help="D-Fire root (YOLO format)")
    parser.add_argument("--hpwren", type=Path, help="HPWREN / AI For Mankind smoke boxes root (YOLO format)")
    parser.add_argument("--out", type=Path, default=Path("data/processed/m2_smoke"))
    parser.add_argument("--val", type=float, default=0.1, help="validation fraction (of groups)")
    parser.add_argument("--test", type=float, default=0.1, help="HPWREN test fraction (of events)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--copy", action="store_true", help="copy images instead of symlinking (Windows)")
    args = parser.parse_args()
    report = build(args.dfire, args.hpwren, args.out, args.val, args.test, args.seed, args.copy)
    print(json.dumps(report, indent=2))
    if not report["leakage"]["leakage_free"]:
        raise SystemExit("LEAKAGE DETECTED: see split_report.json")


if __name__ == "__main__":
    main()

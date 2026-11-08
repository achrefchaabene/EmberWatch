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
import hashlib
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


# ------------------------------------------------------- grouping fallback
def image_dhash(path: Path) -> int:
    """64-bit perceptual fingerprint of an image (same function as the EDA)."""
    import numpy as np
    from PIL import Image

    from src.detection.eda import dhash

    with Image.open(path) as im:
        gray = np.asarray(im.convert("L"), dtype=np.uint8)
    return dhash(gray)


def group_by_similarity(samples: list[Sample], max_hamming: int = 8) -> dict:
    """Give visually near-identical images the same event id (union-find on dHash distance).

    Used when file names carry no event / camera id (e.g. Roboflow exports with random
    ids): frames of the same fixed camera minutes apart are nearly identical and must
    stay in the same split, otherwise the test score is inflated.
    max_hamming: bits that may differ out of 64 (8 = 12.5 %, a common near-duplicate bound).
    """
    hashes = []
    for smp in samples:
        try:
            hashes.append(image_dhash(smp.image))
        except Exception:  # noqa: BLE001 - corrupt image: own group, dropped later anyway
            hashes.append(None)
    parent = list(range(len(samples)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]  # path halving keeps trees flat
            i = parent[i]
        return i

    for i in range(len(samples)):
        if hashes[i] is None:
            continue
        for j in range(i + 1, len(samples)):
            if hashes[j] is not None and (hashes[i] ^ hashes[j]).bit_count() <= max_hamming:
                parent[root(j)] = root(i)
    sizes: Counter = Counter()
    for i, smp in enumerate(samples):
        smp.event = f"simgroup{root(i)}"
        sizes[smp.event] += 1
    largest = max(sizes.values()) if sizes else 0
    return {"method": f"visual similarity (dHash, hamming <= {max_hamming})", "groups": len(sizes),
            "largest_group": largest, "largest_group_share": round(largest / max(len(samples), 1), 3)}


# -------------------------------------------------------------------- splits
def group_split(samples: list[Sample], val: float, test: float, seed: int) -> dict[str, list[Sample]]:
    """Split whole groups (events / look-alike scenes) so no group crosses splits.

    Targets are counted in IMAGES, not in groups: with unequal groups, "10 % of the
    groups" can mean 1 % or 60 % of the images. Groups are visited in a seeded random
    order and placed in test, then val, until each reaches its image target; a group
    that would overshoot a target by more than 50 % goes to train instead (a giant
    look-alike group must never become the whole test set).
    """
    groups: dict[str, list[Sample]] = defaultdict(list)
    for smp in samples:
        groups[smp.event].append(smp)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    total = len(samples)
    targets = {"test": test * total, "val": val * total}
    parts: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    for key in keys:
        size = len(groups[key])
        dest = "train"
        for split in ("test", "val"):
            room = targets[split] - len(parts[split])
            # take the group if the split still needs images and the group does not overshoot too much
            if room > 0 and len(parts[split]) + size <= max(1.5 * targets[split], 1):
                dest = split
                break
        parts[dest] += groups[key]
    return parts


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
def clean_box(row: list[str]) -> list[str] | None:
    """Validate one YOLO row; clip slightly-out-of-range boxes, drop impossible ones."""
    try:
        x, y, w, h = map(float, row[1:5])
    except ValueError:
        return None
    # Convert to corners, clip to the image, convert back (annotation tools often overshoot by 1px).
    x1, y1, x2, y2 = max(0.0, x - w / 2), max(0.0, y - h / 2), min(1.0, x + w / 2), min(1.0, y + h / 2)
    if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
        return None  # zero-area or fully outside: noise for the loss
    return [row[0], f"{(x1 + x2) / 2:.6f}", f"{(y1 + y2) / 2:.6f}", f"{x2 - x1:.6f}", f"{y2 - y1:.6f}"]


def readable(image: Path) -> bool:
    """True if PIL can decode the image (corrupt files crash training workers)."""
    from PIL import Image

    try:
        with Image.open(image) as im:
            im.verify()
        return True
    except Exception:  # noqa: BLE001 - any decode failure means unusable
        return False


def write_image(src: Path, dst: Path, copy: bool, max_side: int | None) -> None:
    """Symlink, copy, or resize-copy one image. YOLO labels are normalised, so resizing keeps them valid."""
    if max_side:
        from PIL import Image

        with Image.open(src) as im:
            if max(im.size) > max_side:
                im = im.convert("RGB")
                im.thumbnail((max_side, max_side), Image.LANCZOS)  # LANCZOS: sharpest downscale
                im.save(dst.with_suffix(".jpg"), quality=95)
                return
    if copy or max_side:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def write_split(split: str, samples: list[Sample], smoke_id: int, out: Path, copy: bool,
                max_side: int | None = None) -> Counter:
    """Write cleaned images + smoke-only labels; return counts (incl. what was dropped)."""
    stats: Counter = Counter(images=0, boxes=0, negatives=0)
    (out / "images" / split).mkdir(parents=True, exist_ok=True)
    (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for s in samples:
        if not readable(s.image):
            stats["dropped_corrupt_images"] += 1
            continue
        name = f"{s.source}__{s.image.stem}"
        dst_img = out / "images" / split / f"{name}{s.image.suffix.lower()}"
        if not dst_img.exists() and not dst_img.with_suffix(".jpg").exists():
            write_image(s.image, dst_img, copy, max_side)
        # Keep only smoke boxes, remap to class 0, clean coordinates.
        smoke_rows = []
        for r in read_labels(s.label):
            if int(r[0]) != smoke_id:
                continue
            fixed = clean_box(["0", *r[1:]])
            if fixed is None:
                stats["dropped_invalid_boxes"] += 1
            else:
                smoke_rows.append(fixed)
        (out / "labels" / split / f"{name}.txt").write_text("".join(" ".join(r) + "\n" for r in smoke_rows))
        stats["images"] += 1
        stats["boxes"] += len(smoke_rows)
        stats["negatives"] += int(not smoke_rows)
    return stats


def file_md5(path: Path) -> str:
    """Content fingerprint: two files with the same bytes = the same image, whatever their names."""
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def drop_cross_split_duplicates(merged: dict[str, list]) -> dict[str, int]:
    """Remove from val/test any image whose CONTENT already exists in an earlier split.

    Names and event ids cannot catch a copied file; a byte-identical image in test
    means the model was trained on its own test image (inflated score).
    Order of priority: train keeps everything, then val, then test.
    """
    seen: set[str] = set()
    removed = {"val": 0, "test": 0}
    for split in ("train", "val", "test"):
        kept = []
        for sample, sid in merged[split]:
            digest = file_md5(sample.image)
            if split != "train" and digest in seen:
                removed[split] += 1
                continue
            seen.add(digest)
            kept.append((sample, sid))
        merged[split] = kept
    return removed


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


def build(dfire: Path | None, hpwren: Path | None, out: Path, val: float, test: float, seed: int, copy: bool,
          max_side: int | None = None, similarity_hamming: int = 8) -> dict:
    """Build the merged dataset and return the report written to split_report.json."""
    if not dfire and not hpwren:
        raise ValueError("Give at least one source (--dfire and/or --hpwren).")
    out.mkdir(parents=True, exist_ok=True)
    merged: dict[str, list[tuple[Sample, int]]] = {"train": [], "val": [], "test": []}
    report: dict = {"seed": seed, "sources": {}}

    if dfire:
        samples = collect_samples(dfire, "dfire")
        smoke_id, per_class = detect_smoke_class(samples)
        yaml_id = smoke_class_from_yaml(dfire)
        if yaml_id is not None and yaml_id != smoke_id:
            # Two independent sources disagree: stop instead of training a fire detector by mistake.
            raise SystemExit(f"D-Fire smoke class mismatch: data.yaml says {yaml_id}, counts say {smoke_id}. "
                             "Open the data.yaml and a few label files to decide.")
        report["sources"]["dfire"] = {
            "images": len(samples),
            "smoke_class_id": smoke_id,
            "smoke_class_id_from_yaml": yaml_id,
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
        report["sources"]["hpwren"] = {"images": len(samples), "smoke_class_id": smoke_id}
        if len({smp.event for smp in samples}) == len(samples):
            # Every name is unique -> names hold no event id -> group frames by how they look.
            report["sources"]["hpwren"]["grouping"] = group_by_similarity(samples, similarity_hamming)
        else:
            report["sources"]["hpwren"]["grouping"] = {"method": "event id from file names",
                                                       "groups": len({smp.event for smp in samples})}
        for k, v in group_split(samples, val, test, seed).items():
            merged[k] += [(s, smoke_id) for s in v]

    report["removed_cross_split_duplicates"] = drop_cross_split_duplicates(merged)
    report["splits"] = {}
    for split, pairs in merged.items():
        stats = Counter(images=0, boxes=0, negatives=0)
        for s, sid in pairs:
            stats.update(write_split(split, [s], sid, out, copy, max_side))  # update keeps zero counts
        report["splits"][split] = dict(stats)

    report["per_source_splits"] = {split: dict(Counter(smp.source for smp, _ in pairs)) for split, pairs in merged.items()}
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
    parser.add_argument("--max-side", type=int, default=None,
                        help="resize images whose long side is larger (only if the EDA recommends it)")
    parser.add_argument("--similarity-hamming", type=int, default=8,
                        help="max differing bits (of 64) to group look-alike HPWREN frames")
    args = parser.parse_args()
    report = build(args.dfire, args.hpwren, args.out, args.val, args.test, args.seed, args.copy, args.max_side,
                   args.similarity_hamming)
    print(json.dumps(report, indent=2))
    if not report["leakage"]["leakage_free"]:
        raise SystemExit("LEAKAGE DETECTED: see split_report.json")


if __name__ == "__main__":
    main()

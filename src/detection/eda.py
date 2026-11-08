"""Exploratory data analysis of a YOLO dataset -> data-driven preprocessing decisions.

Every measurement exists because it drives a decision:

| Measurement                         | Decision it drives                                        |
|-------------------------------------|-----------------------------------------------------------|
| corrupt images, invalid boxes       | drop / fix before training (garbage in, garbage out)      |
| image size distribution             | offline resize for speed? training imgsz?                 |
| box size in pixels (small < 32 px)  | SAHI tiling at inference, larger imgsz ablation           |
| boxes per image, negatives share    | need more hard negatives (clouds, fog)?                   |
| grayscale share (IR / mono cameras) | ToGray augmentation probability                           |
| brightness (night / dusk share)     | brightness / contrast augmentation range                  |
| sharpness (Laplacian variance)      | add blur augmentation or not                              |
| near-duplicates ACROSS splits       | data leakage -> inflated test score                       |
| box centre heat-map                 | where smoke appears (above horizon) -> crop / tiling ideas|

Usage:
    python -m src.detection.eda --data data/processed/m2_smoke --out artifacts/eda_m2
    python -m src.detection.eda --data data/raw/dfire --out artifacts/eda_dfire_raw
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLIT_ALIASES = {"train": "train", "valid": "val", "val": "val", "test": "test"}
SMALL_PX, MEDIUM_PX = 32, 96  # COCO object-size convention (sqrt of box area in pixels)
DARK_MEAN = 60  # mean grey level below which a frame is "night / dusk"
BLURRY_LAPVAR = 100  # Laplacian variance below which a frame is blurry (common heuristic)


# ------------------------------------------------------------------ discovery
def find_pairs(root: Path) -> list[tuple[str, Path, Path | None]]:
    """Return (split, image, label) for every image under any '<split>/images' or 'images/<split>' layout."""
    pairs = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or "images" not in img.parts:
            continue
        i = len(img.parts) - 1 - img.parts[::-1].index("images")
        label = Path(*img.parts[:i], "labels", *img.parts[i + 1:]).with_suffix(".txt")
        # Split name is either the folder before 'images' (raw) or the one after it (processed).
        before = img.parts[i - 1].lower() if i > 0 else ""
        after = img.parts[i + 1].lower() if i + 1 < len(img.parts) - 1 else ""
        split = SPLIT_ALIASES.get(before) or SPLIT_ALIASES.get(after) or "all"
        pairs.append((split, img, label if label.exists() else None))
    return pairs


# ------------------------------------------------------------------ per-image measurements
def dhash(gray: np.ndarray, size: int = 8) -> int:
    """Difference hash: 64-bit fingerprint robust to resize / JPEG -> finds near-duplicates."""
    small = np.asarray(Image.fromarray(gray).resize((size + 1, size), Image.BILINEAR), dtype=np.int16)
    bits = (small[:, 1:] > small[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


def laplacian_var(gray: np.ndarray) -> float:
    """Sharpness: variance of the Laplacian (low = blurry). Pure numpy, no OpenCV needed."""
    g = gray.astype(np.float32)
    lap = -4 * g[1:-1, 1:-1] + g[:-2, 1:-1] + g[2:, 1:-1] + g[1:-1, :-2] + g[1:-1, 2:]
    return float(lap.var())


def image_stats(path: Path, max_side: int = 512) -> dict | None:
    """Open one image and measure it on a downscaled copy (fast, enough for statistics)."""
    try:
        with Image.open(path) as im:
            w, h = im.size
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side))
            rgb = np.asarray(im, dtype=np.int16)
    # Broad on purpose: once Ultralytics is imported it patches Image.open and a corrupt
    # file can raise ModuleNotFoundError (pi_heif) instead of UnidentifiedImageError.
    except Exception:  # noqa: BLE001
        return None
    gray = rgb.mean(axis=2).astype(np.uint8)
    # Grayscale if colour channels are almost identical (mono / IR cameras).
    chroma = float(np.abs(rgb[..., 0] - rgb[..., 1]).mean() + np.abs(rgb[..., 1] - rgb[..., 2]).mean())
    return {
        "w": w, "h": h,
        "mean": float(gray.mean()), "contrast": float(gray.std()),
        "lapvar": laplacian_var(gray), "grayscale": chroma < 2.0,
        "hash": dhash(gray),
    }


def parse_labels(label: Path | None) -> tuple[list[tuple[int, float, float, float, float]], int]:
    """Return valid boxes and the number of invalid rows (bad format, out of [0,1], zero area)."""
    if label is None:
        return [], 0
    boxes, invalid = [], 0
    for row in label.read_text().splitlines():
        parts = row.split()
        if not parts:
            continue
        try:
            c, x, y, w, h = int(parts[0]), *map(float, parts[1:5])
        except (ValueError, IndexError):
            invalid += 1
            continue
        if len(parts) != 5 or w <= 0 or h <= 0 or not all(0 <= v <= 1 for v in (x, y, w, h)):
            invalid += 1
            continue
        boxes.append((c, x, y, w, h))
    return boxes, invalid


# ------------------------------------------------------------------ analysis
def analyse(root: Path, max_images: int | None = None, seed: int = 0) -> dict:
    """Measure the dataset. max_images samples images uniformly for speed on big datasets."""
    pairs = find_pairs(root)
    if max_images and len(pairs) > max_images:
        rng = np.random.default_rng(seed)
        pairs = [pairs[i] for i in sorted(rng.choice(len(pairs), max_images, replace=False))]

    per_split: dict[str, Counter] = defaultdict(Counter)
    sizes, means, contrasts, lapvars = [], [], [], []
    box_px, box_rel_area, box_aspect, boxes_per_img, centres = [], [], [], [], []
    class_images: Counter = Counter()
    hashes: dict[int, list[tuple[str, str]]] = defaultdict(list)  # hash -> [(split, file)]
    gray_n = corrupt = invalid_rows = 0

    for split, img, label in pairs:
        st = image_stats(img)
        if st is None:
            corrupt += 1
            continue
        boxes, bad = parse_labels(label)
        invalid_rows += bad
        per_split[split]["images"] += 1
        per_split[split]["boxes"] += len(boxes)
        per_split[split]["negatives"] += int(not boxes)
        sizes.append((st["w"], st["h"]))
        means.append(st["mean"])
        contrasts.append(st["contrast"])
        lapvars.append(st["lapvar"])
        gray_n += st["grayscale"]
        hashes[st["hash"]].append((split, img.name))
        boxes_per_img.append(len(boxes))
        class_images.update({c for c, *_ in boxes})
        for _, x, y, w, h in boxes:
            box_px.append(float(np.sqrt(w * st["w"] * h * st["h"])))  # side of the equivalent square
            box_rel_area.append(w * h)
            box_aspect.append((w * st["w"]) / max(h * st["h"], 1e-6))
            centres.append((x, y))

    n = len(sizes)
    if n == 0:
        raise ValueError(f"No readable image found under {root}")
    px = np.array(box_px) if box_px else np.zeros(1)
    long_side = np.array([max(s) for s in sizes])
    cross_groups = [g for g in hashes.values() if len({sp for sp, _ in g}) > 1]
    cross_split_dups = len(cross_groups)
    stats = {
        "root": str(root),
        "images_analysed": n,
        "corrupt_images": corrupt,
        "invalid_label_rows": invalid_rows,
        "splits": {k: dict(v) for k, v in per_split.items()},
        "images_per_class": {str(k): v for k, v in sorted(class_images.items())},
        "image_long_side": _describe(long_side),
        "boxes": {
            "total": len(box_px),
            "side_px": _describe(px),
            "small_share": round(float((px < SMALL_PX).mean()), 3) if box_px else 0.0,
            "medium_share": round(float(((px >= SMALL_PX) & (px < MEDIUM_PX)).mean()), 3) if box_px else 0.0,
            "large_share": round(float((px >= MEDIUM_PX).mean()), 3) if box_px else 0.0,
            "relative_area": _describe(np.array(box_rel_area) if box_rel_area else np.zeros(1)),
            "aspect_w_over_h": _describe(np.array(box_aspect) if box_aspect else np.zeros(1)),
            "per_image": _describe(np.array(boxes_per_img)),
        },
        "negatives_share": round(sum(b == 0 for b in boxes_per_img) / n, 3),
        "grayscale_share": round(gray_n / n, 3),
        "dark_share": round(float((np.array(means) < DARK_MEAN).mean()), 3),
        "brightness_mean": _describe(np.array(means)),
        "contrast_std": _describe(np.array(contrasts)),
        "blurry_share": round(float((np.array(lapvars) < BLURRY_LAPVAR).mean()), 3),
        "sharpness_lapvar": _describe(np.array(lapvars)),
        "near_duplicate_groups_across_splits": cross_split_dups,
        # First groups to open side by side (same perceptual hash, different splits).
        "near_duplicate_examples": [[f"{sp}/{name}" for sp, name in g[:4]] for g in cross_groups[:10]],
        "box_centres": centres,  # kept for the heat-map, removed from the JSON
    }
    stats["recommendations"] = recommend(stats)
    return stats


def _describe(a: np.ndarray) -> dict:
    """Compact distribution summary."""
    return {k: round(float(v), 4) for k, v in {
        "min": a.min(), "p10": np.percentile(a, 10), "median": np.median(a),
        "mean": a.mean(), "p90": np.percentile(a, 90), "max": a.max()}.items()}


# ------------------------------------------------------------------ decisions
def recommend(s: dict) -> list[dict]:
    """Turn measurements into explicit decisions (rule, evidence, action)."""
    r = []

    def add(topic, evidence, action):
        r.append({"topic": topic, "evidence": evidence, "action": action})

    if s["corrupt_images"] or s["invalid_label_rows"]:
        add("cleaning", f"{s['corrupt_images']} corrupt images, {s['invalid_label_rows']} invalid label rows",
            "prepare_dataset drops corrupt images and invalid boxes (already implemented).")
    small = s["boxes"]["small_share"]
    if small >= 0.3:
        add("small objects", f"{small:.0%} of boxes are < {SMALL_PX}px",
            "Keep SAHI tiling at inference; run an ablation with imgsz=960 or 1024 for training.")
    else:
        add("small objects", f"{small:.0%} of boxes are < {SMALL_PX}px",
            "imgsz=640 is enough for training; SAHI still needed on full-resolution tower frames.")
    med_long = s["image_long_side"]["median"]
    if med_long > 1600:
        add("resize", f"median long side = {med_long:.0f}px",
            "Pre-resize offline to max side 1280 (prepare_dataset --max-side 1280) to speed up loading; "
            "keep originals for inference/SAHI.")
    else:
        add("resize", f"median long side = {med_long:.0f}px",
            "No offline resize: YOLO letterboxes on the fly; resizing would only lose detail.")
    g = s["grayscale_share"]
    add("grayscale", f"{g:.0%} grayscale images (mono / IR cameras)",
        f"ToGray augmentation with p={max(0.05, round(g, 2))} so the model does not rely on colour only.")
    d = s["dark_share"]
    add("lighting", f"{d:.0%} dark frames (mean grey < {DARK_MEAN})",
        "RandomBrightnessContrast(±0.2) + hsv_v=0.4." + (" Night frames are rare: report night as a known failure mode (P4)." if d < 0.05 else ""))
    b = s["blurry_share"]
    add("blur", f"{b:.0%} blurry frames (Laplacian var < {BLURRY_LAPVAR})",
        "No extra blur augmentation (data already blurry; blur erases thin smoke)." if b >= 0.2
        else "Light blur augmentation only (p=0.05, kernel 3): camera focus / haze varies.")
    neg = s["negatives_share"]
    add("negatives", f"{neg:.0%} images without smoke",
        "Add hard negatives (clouds, fog, haze; FIgLib pre-ignition frames) in Sprint 2 (M4)." if neg < 0.2
        else "Enough negatives for the baseline; M4 will add targeted hard negatives.")
    dup = s["near_duplicate_groups_across_splits"]
    add("leakage", f"{dup} near-duplicate groups shared between splits",
        "Open the files in eda.json -> near_duplicate_examples; if they are the same scene, "
        "remove them from val/test before reporting metrics." if dup
        else "No near-duplicates across splits: test score is trustworthy.")
    return r


# ------------------------------------------------------------------ outputs
def save_report(stats: dict, out: Path) -> None:
    """Write eda.json, figures and a short markdown report."""
    out.mkdir(parents=True, exist_ok=True)
    centres = stats.pop("box_centres")
    (out / "eda.json").write_text(json.dumps(stats, indent=2))
    _figures(stats, centres, out)
    lines = [f"# Dataset study — `{stats['root']}`", "",
             f"Images analysed: **{stats['images_analysed']}** · boxes: **{stats['boxes']['total']}** · "
             f"corrupt: {stats['corrupt_images']} · invalid label rows: {stats['invalid_label_rows']}", "",
             "| Split | Images | Boxes | Negatives |", "|---|---|---|---|"]
    for k, v in stats["splits"].items():
        lines.append(f"| {k} | {v.get('images', 0)} | {v.get('boxes', 0)} | {v.get('negatives', 0)} |")
    lines += ["", "## Decisions", "", "| Topic | Evidence | Action |", "|---|---|---|"]
    lines += [f"| {r['topic']} | {r['evidence']} | {r['action']} |" for r in stats["recommendations"]]
    lines += ["", "![box sizes](box_sizes.png)", "![box centres](box_centres.png)", "![image quality](image_quality.png)"]
    (out / "eda.md").write_text("\n".join(lines), encoding="utf-8")


def _figures(stats: dict, centres: list, out: Path) -> None:
    """Three figures: box sizes, where smoke is in the frame, image quality."""
    import matplotlib
    matplotlib.use("Agg")  # no display needed (Kaggle / CI)
    import matplotlib.pyplot as plt

    b = stats["boxes"]
    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.bar(["small\n<32px", "medium\n32-96px", "large\n>96px"],
           [b["small_share"], b["medium_share"], b["large_share"]], color="#C2410C")
    ax.set_ylabel("share of boxes"); ax.set_title("Smoke box sizes (COCO convention)")
    fig.tight_layout(); fig.savefig(out / "box_sizes.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    if centres:
        xy = np.array(centres)
        ax.hist2d(xy[:, 0], xy[:, 1], bins=20, range=[[0, 1], [0, 1]], cmap="Oranges")
    ax.invert_yaxis(); ax.set_xlabel("x (relative)"); ax.set_ylabel("y (relative, top = 0)")
    ax.set_title("Where smoke boxes are")
    fig.tight_layout(); fig.savefig(out / "box_centres.png", dpi=110); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, (key, title) in zip(axes, [("brightness_mean", "brightness (median)"),
                                       ("contrast_std", "contrast (median)"),
                                       ("sharpness_lapvar", "sharpness (median)")]):
        d = stats[key]
        ax.bar(["p10", "median", "p90"], [d["p10"], d["median"], d["p90"]], color="#1E3A5F")
        ax.set_title(title)
    fig.tight_layout(); fig.savefig(out / "image_quality.png", dpi=110); plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True, help="dataset root (raw or processed)")
    parser.add_argument("--out", type=Path, default=Path("artifacts/eda"))
    parser.add_argument("--max-images", type=int, default=5000, help="uniform sample for speed (0 = all)")
    args = parser.parse_args()
    stats = analyse(args.data, args.max_images or None)
    save_report(stats, args.out)
    for r in stats["recommendations"]:
        print(f"- [{r['topic']}] {r['evidence']} -> {r['action']}")
    print(f"Report: {args.out / 'eda.md'}")


if __name__ == "__main__":
    main()

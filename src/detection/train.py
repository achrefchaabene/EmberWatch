"""Train the M2 YOLO smoke detector from configs/m2_yolo.yaml (run on Kaggle / Colab GPU).

Usage:
    python -m src.detection.train --data data/processed/m2_smoke/data.yaml
    python -m src.detection.train --data ... --epochs 3 --fraction 0.05   # quick smoke test
    python -m src.detection.train --resume runs/m2/yolo_smoke/weights/last.pt   # after a Kaggle cut
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "m2_yolo.yaml"


def load_config(path: Path, overrides: dict) -> dict:
    """Read the YAML hyperparameters, then apply non-empty CLI overrides."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def build_albumentations(spec: dict | None) -> list | None:
    """Our pixel-level augmentations, replacing Ultralytics' defaults (blur/gray/CLAHE at 1 %).

    Returns None if albumentations is not installed (Ultralytics then uses its defaults).
    Every transform keeps box coordinates unchanged (pixel-level only).
    """
    if not spec:
        return None
    try:
        import albumentations as A
    except ImportError:
        print("albumentations not installed: Ultralytics default pixel augmentations are used.")
        return None
    transforms = [
        A.ToGray(p=spec.get("to_gray", 0.0)),  # mono / IR cameras
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=spec.get("brightness_contrast", 0.0)),
        A.ImageCompression(quality_range=(60, 100), p=spec.get("jpeg_compression", 0.0)),  # streamed JPEG
        A.GaussNoise(std_range=(0.02, 0.06), p=spec.get("gauss_noise", 0.0)),  # light sensor noise only
        A.Blur(blur_limit=(3, 3), p=spec.get("blur", 0.0)),  # kernel 3 max: thin smoke must survive
        A.CLAHE(clip_limit=2.0, p=spec.get("clahe", 0.0)),  # off by default (ablation)
    ]
    return [t for t in transforms if t.p > 0]


def train(data: Path, cfg: dict, project: Path, name: str) -> Path:
    """Fine-tune YOLO with the given hyperparameters; return the path of best.pt."""
    from ultralytics import YOLO

    cfg = dict(cfg)
    model_name = cfg.pop("model")
    augmentations = build_albumentations(cfg.pop("albumentations", None))
    yolo = YOLO(model_name)
    extra = {"augmentations": augmentations} if augmentations else {}
    yolo.train(data=str(data), project=str(project), name=name, exist_ok=True, plots=True, **cfg, **extra)

    run_dir = Path(yolo.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    # Versioned record of exactly what was trained (report, model card, reproducibility).
    record = {"data": str(data), "model": model_name, "best_weights": str(best), "hyperparameters": cfg,
              "albumentations": [type(t).__name__ + f"(p={t.p})" for t in (augmentations or [])]}
    (run_dir / "run_config.json").write_text(json.dumps(record, indent=2, default=str))
    return best


def resume(last: Path) -> Path:
    """Continue an interrupted run from last.pt (same hyperparameters, same LR schedule)."""
    from ultralytics import YOLO

    yolo = YOLO(str(last))
    yolo.train(resume=True)
    return Path(yolo.trainer.save_dir) / "weights" / "best.pt"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, help="data.yaml produced by prepare_dataset")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--resume", type=Path, help="path to last.pt of an interrupted run")
    parser.add_argument("--project", type=Path, default=Path("runs/m2"))
    parser.add_argument("--name", default="yolo_smoke")
    # Optional overrides of the YAML (None = keep the YAML value).
    parser.add_argument("--model")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--batch", type=int)
    parser.add_argument("--device")
    parser.add_argument("--fraction", type=float, help="use a fraction of train images (quick tests)")
    parser.add_argument("--workers", type=int)
    args = parser.parse_args()

    if args.resume:
        print(f"Best weights: {resume(args.resume)}")
        return
    if not args.data:
        parser.error("--data is required unless --resume is given")
    overrides = {k: getattr(args, k) for k in ("model", "epochs", "imgsz", "batch", "device", "fraction", "workers")}
    cfg = load_config(args.config, overrides)
    print(f"Best weights: {train(args.data, cfg, args.project, args.name)}")


if __name__ == "__main__":
    main()

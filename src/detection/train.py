"""Train the M2 YOLO smoke detector (run on Kaggle / Colab GPU).

Usage:
    python -m src.detection.train --data data/processed/m2_smoke/data.yaml \
        --model yolov8s.pt --epochs 50 --imgsz 640 --batch 16 --device 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def train(data: Path, model: str, epochs: int, imgsz: int, batch: int, device: str,
          project: Path, name: str, seed: int, patience: int) -> Path:
    """Fine-tune a pretrained YOLO on the smoke dataset and return best.pt."""
    from ultralytics import YOLO

    yolo = YOLO(model)  # 'yolov8s.pt' = COCO-pretrained, 'yolov8n.yaml' = from scratch
    yolo.train(
        data=str(data),
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=str(project),
        name=name,
        seed=seed,  # reproducibility (CDC non-functional requirement)
        deterministic=True,
        patience=patience,  # early stopping on val mAP
        # Augmentation: mosaic is on by default; copy_paste adds smoke onto clean scenes.
        mosaic=1.0,
        copy_paste=0.1,
        fliplr=0.5,
        flipud=0.0,  # smoke rises: never flip upside-down
        exist_ok=True,
        plots=True,
    )
    run_dir = Path(yolo.trainer.save_dir)
    best = run_dir / "weights" / "best.pt"
    # Keep a small, versioned record of the run next to the weights.
    (run_dir / "run_config.json").write_text(json.dumps({
        "data": str(data), "base_model": model, "epochs": epochs, "imgsz": imgsz,
        "batch": batch, "seed": seed, "best_weights": str(best),
    }, indent=2))
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="0", help="'0' for GPU 0, 'cpu' for CPU")
    parser.add_argument("--project", type=Path, default=Path("runs/m2"))
    parser.add_argument("--name", default="yolo_smoke")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--patience", type=int, default=15)
    args = parser.parse_args()
    best = train(args.data, args.model, args.epochs, args.imgsz, args.batch, args.device,
                 args.project, args.name, args.seed, args.patience)
    print(f"Best weights: {best}")


if __name__ == "__main__":
    main()

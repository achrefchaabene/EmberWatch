"""Evaluate the M2 detector: mAP@0.5 on the test split + throughput (CDC M2 acceptance).

Writes artifacts/m2_metrics.json, read by the report and the dashboard.

Usage:
    python -m src.detection.evaluate --weights runs/m2/yolo_smoke/weights/best.pt \
        --data data/processed/m2_smoke/data.yaml --device 0
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import numpy as np

MVP_MAP50 = 0.55  # CDC §11 MVP target
STRETCH_MAP50 = 0.65  # CDC §11 stretch target


def evaluate_map(weights: Path, data: Path, imgsz: int, device: str) -> dict:
    """Run Ultralytics validation on the TEST split (never on val, to avoid optimism)."""
    from ultralytics import YOLO

    metrics = YOLO(str(weights)).val(data=str(data), split="test", imgsz=imgsz, device=device, plots=False, verbose=False)
    return {
        "map50": round(float(metrics.box.map50), 4),
        "map50_95": round(float(metrics.box.map), 4),
        "precision": round(float(metrics.box.mp), 4),
        "recall": round(float(metrics.box.mr), 4),
    }


def benchmark(weights: Path, images: list[Path], device: str, use_tiling: bool) -> dict:
    """Mean latency per frame (ms) with the real inference path used by the API."""
    from PIL import Image

    from src.detection.detector import SmokeDetector

    detector = SmokeDetector(weights, use_tiling=use_tiling, device=device)
    frames = [np.asarray(Image.open(p).convert("RGB")) for p in images]
    detector.predict(frames[0])  # warm-up (first call loads kernels)
    times = []
    for frame in frames:
        t0 = time.perf_counter()
        detector.predict(frame)
        times.append((time.perf_counter() - t0) * 1000)
    return {"tiling": use_tiling, "frames": len(frames), "ms_per_frame": round(float(np.mean(times)), 1)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="0")
    parser.add_argument("--bench-images", type=int, default=20, help="test images used for latency")
    parser.add_argument("--out", type=Path, default=Path("artifacts/m2_metrics.json"))
    args = parser.parse_args()

    result = {"weights": args.weights.name, "dataset": str(args.data), "split": "test"}
    result.update(evaluate_map(args.weights, args.data, args.imgsz, args.device))
    result["meets_mvp"] = result["map50"] >= MVP_MAP50
    result["meets_stretch"] = result["map50"] >= STRETCH_MAP50

    test_dir = args.data.parent / "images" / "test"
    images = sorted(test_dir.glob("*"))[: args.bench_images]
    if images:
        result["throughput"] = [benchmark(args.weights, images, args.device, t) for t in (False, True)]
        result["hardware"] = {"device": args.device, "machine": platform.processor() or platform.machine()}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

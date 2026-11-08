# M2 — Frame-level smoke detector

Owner: **Mouhib** (Computer Vision Lead) · Sprint 1 (weeks 2–3) · CDC module M2

## What it does

Detects **smoke** on a single tower-camera frame and returns boxes in original-frame
pixels, following the data contract (CDC §8). It is the baseline for the
spatio-temporal detector M3, which will decide alerts over several frames.

```
frame (any resolution)
  └─ SAHI: overlapping 640 px tiles + one full-frame pass
       └─ YOLO (smoke class) on each tile
            └─ merge boxes (NMS, IOS 0.5) → boxes in frame pixels
                 └─ build_alert(): strongest box → Alert (PENDING_CONFIRMATION)
```

## Files

| File | Role |
|---|---|
| `configs/datasets.yaml` + `src/detection/download.py` | Every dataset link in one place; download from Kaggle, Roboflow, HTTP, Drive, GitHub, OneDrive |
| `src/detection/eda.py` | Dataset study → data-driven preprocessing / augmentation decisions |
| `configs/m2_yolo.yaml` | Hyperparameters and augmentations, each one explained |
| `src/detection/contract.py` | `Detection`, `Alert` (CDC §8.1), tile ids on a 224 px grid |
| `src/detection/prepare_dataset.py` | D-Fire + HPWREN → smoke-only YOLO dataset, event-level split, leakage report |
| `src/detection/train.py` | YOLO fine-tuning from `configs/m2_yolo.yaml`, custom Albumentations, `--resume` |
| `src/detection/evaluate.py` | mAP@0.5 on the **test** split + latency with/without tiling → `artifacts/m2_metrics.json` |
| `src/detection/detector.py` | `SmokeDetector`: YOLO + SAHI inference used by the API |
| `src/api/detection_routes.py` | `POST /api/v1/detect`, `GET /api/v1/alerts` |
| `notebooks/02_m2_train_yolo_kaggle.ipynb` | End-to-end GPU training on Kaggle |
| `tests/test_detection_*.py`, `tests/test_prepare_dataset.py` | Contract, dataset builder and API tests (no GPU / weights needed) |

## Design choices

Full reasoning (filters used / refused, hyperparameters): [M2_DONNEES_ET_ENTRAINEMENT.md](M2_DONNEES_ET_ENTRAINEMENT.md).


- **Smoke only.** D-Fire images with fire but no smoke are kept as negatives (empty labels).
- **Smoke class id is inferred, not hard-coded.** The D-Fire README gives image counts
  (smoke in ~10.5k images, fire in ~5.8k) but not the numeric ids; the class present in
  more images is smoke. Check `split_report.json → sources.dfire.images_per_class`.
- **No leakage.** HPWREN frames are grouped by fire event (`<event>__<timestamp>` file
  names) before splitting. D-Fire keeps its official test split for comparability.
  `prepare_dataset` exits with an error if any event or image crosses splits.
- **Test split for reported numbers.** Validation is only used for early stopping.
- **No vertical flip / rotation / mixup** in augmentation: smoke rises, tower cameras are levelled, blended images look like smoke.
- **copy_paste is not used**: Ultralytics ignores it without segmentation masks (boxes only here).
- **Human in the loop.** Every alert is `PENDING_CONFIRMATION` (principle P1).

## How to run

```bash
# 1. Build the dataset (add --copy on Windows instead of symlinks)
python -m src.detection.prepare_dataset --dfire data/raw/dfire --hpwren data/raw/hpwren_smoke --out data/processed/m2_smoke

# 2. Train on GPU (or run notebooks/02_m2_train_yolo_kaggle.ipynb on Kaggle)
python -m src.detection.train --data data/processed/m2_smoke/data.yaml --model yolov8s.pt --device 0

# 3. Evaluate on the test split
python -m src.detection.evaluate --weights runs/m2/yolo_smoke/weights/best.pt --data data/processed/m2_smoke/data.yaml --device 0

# 4. Serve: copy best.pt to artifacts/m2_smoke_best.pt (or set EMBERWATCH_DETECTOR_WEIGHTS)
python -m uvicorn src.api.main:app --reload
curl -F "image=@frame.jpg" -F "camera_id=CAM-03" http://127.0.0.1:8000/api/v1/detect

# Tests
python -m pytest tests -q
```

Environment variables: `EMBERWATCH_DETECTOR_WEIGHTS`, `EMBERWATCH_DETECTOR_CONF` (default 0.25),
`EMBERWATCH_DEVICE` (`cpu`, `cuda:0`).

## Acceptance (CDC §11)

| Metric | MVP | Stretch | Result |
|---|---|---|---|
| Smoke mAP@0.5 (test) | ≥ 0.55 | ≥ 0.65 | _fill after Kaggle run_ |
| Frame precision / recall on FIgLib test | reported | — | _Sprint 1, week 3_ |
| Throughput GPU / CPU | reported | — | `artifacts/m2_metrics.json` |

## Licence note

Ultralytics YOLO is AGPL-3.0. Fine for this academic project; a commercial reuse by
Tradrly would need an Ultralytics enterprise licence or a permissive alternative
(e.g. RT-DETR, Apache-2.0). To confirm with the supervisor.

## Next (Sprint 2 — M3)

1. Persistence rule: alert only if smoke is found in k of the last n frames at a consistent location.
2. Cache frozen-backbone tile embeddings on the 224 px grid used by `box_to_tiles`.
3. Temporal model (CNN → temporal transformer → ViT), then frame differences.

"""M2 frame-level smoke detector: YOLO + SAHI tiled inference.

Why tiling: tower-camera panoramas are high resolution (e.g. 3072x2048) and early
smoke is a few dozen pixels wide. Resizing the whole frame to 640 px erases it.
SAHI cuts the frame into overlapping tiles, runs YOLO on each tile, then merges the
boxes back into original-frame pixels (the contract's box convention).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.detection.contract import Detection


class SmokeDetector:
    """Load YOLO weights once, then detect smoke on full frames."""

    def __init__(
        self,
        weights: str | Path,
        conf: float = 0.25,
        slice_size: int = 640,
        overlap: float = 0.2,
        use_tiling: bool = True,
        device: str = "cpu",
    ) -> None:
        # Imported here so modules that only need the contract don't load torch.
        from sahi import AutoDetectionModel

        self.weights = Path(weights)
        self.conf = conf
        self.slice_size = slice_size
        self.overlap = overlap
        self.use_tiling = use_tiling
        self.model_version = f"yolo-smoke@{self.weights.stem}"
        self._model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=str(self.weights),
            confidence_threshold=conf,
            device=device,
        )

    def predict(self, image: np.ndarray) -> list[Detection]:
        """Detect smoke on one RGB frame (H x W x 3, uint8). Boxes are in frame pixels."""
        from sahi.predict import get_prediction, get_sliced_prediction

        h, w = image.shape[:2]
        if self.use_tiling and max(h, w) > self.slice_size:
            result = get_sliced_prediction(
                image,
                self._model,
                slice_height=self.slice_size,
                slice_width=self.slice_size,
                overlap_height_ratio=self.overlap,
                overlap_width_ratio=self.overlap,
                perform_standard_pred=True,  # also run on the full frame for large plumes
                postprocess_type="NMS",
                postprocess_match_metric="IOS",  # merges a plume split across two tiles
                postprocess_match_threshold=0.5,
                verbose=0,
            )
        else:
            result = get_prediction(image, self._model)  # small frame: no need to tile

        detections = []
        for pred in result.object_prediction_list:
            if pred.category.name.lower() != "smoke":
                continue
            x1, y1, x2, y2 = pred.bbox.to_xyxy()
            if x2 <= x1 or y2 <= y1:
                continue  # degenerate box after merging
            detections.append(Detection(box=[x1, y1, x2, y2], confidence=float(pred.score.value)))
        return sorted(detections, key=lambda d: d.confidence, reverse=True)

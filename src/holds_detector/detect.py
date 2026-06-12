"""YOLO segmentation inference: image → per-hold polygons at original resolution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

DEFAULT_CONF = 0.35
DEFAULT_IMGSZ = 1280

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "models"


@dataclass
class RawDetection:
    """One model detection: polygon (N,2) in original-image pixels + confidence."""

    polygon: np.ndarray
    confidence: float


def default_model_path() -> Path:
    """Fine-tuned weights if present, else the pretrained checkpoint
    (auto-downloaded by ultralytics on first use)."""
    best = _MODELS_DIR / "best.pt"
    return best if best.exists() else _MODELS_DIR / "yolo11n-seg.pt"


def run_detection(
    image_path: str | Path,
    model_path: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | None = None,
) -> list[RawDetection]:
    """Run segmentation on one image. Ultralytics handles the resize to
    `imgsz` internally and scales masks back: `masks.xy` polygons are already
    in original-image pixel coordinates."""
    from ultralytics import YOLO  # deferred: heavy import, tests stub this module

    model = YOLO(str(model_path or default_model_path()))
    result = model.predict(str(image_path), conf=conf, imgsz=imgsz, device=device, verbose=False)[0]

    detections: list[RawDetection] = []
    if result.masks is None:
        return detections
    confidences = result.boxes.conf.tolist()
    for polygon, confidence in zip(result.masks.xy, confidences, strict=True):
        if len(polygon) < 3:  # degenerate sliver mask
            continue
        detections.append(RawDetection(polygon=np.asarray(polygon), confidence=float(confidence)))
    return detections

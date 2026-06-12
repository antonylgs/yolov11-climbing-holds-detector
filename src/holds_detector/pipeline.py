"""Orchestration: image → detections → per-hold color → schema + debug image."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np

from holds_detector.color import DISPLAY_BGR, extract_color
from holds_detector.detect import (
    DEFAULT_CONF,
    DEFAULT_IMGSZ,
    RawDetection,
    default_model_path,
    run_detection,
)
from holds_detector.schema import DetectionOutput, Hold

POLYGON_SIMPLIFY_EPS = 1.5  # px; trims mask polygons without visible loss

DetectFn = Callable[[Path], list[RawDetection]]


def _polygon_to_mask(polygon: np.ndarray, height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.round(polygon).astype(np.int32)], 1)
    return mask


def _build_hold(
    hold_id: int, det: RawDetection, image_bgr: np.ndarray, mask: np.ndarray
) -> Hold:
    height, width = image_bgr.shape[:2]
    simplified = cv2.approxPolyDP(
        det.polygon.astype(np.float32), POLYGON_SIMPLIFY_EPS, closed=True
    ).reshape(-1, 2)
    poly_int = np.clip(np.round(simplified), [0, 0], [width - 1, height - 1]).astype(int)

    x1, y1 = poly_int.min(axis=0)
    x2, y2 = poly_int.max(axis=0)
    moments = cv2.moments(mask, binaryImage=True)
    if moments["m00"] > 0:
        center = [int(moments["m10"] / moments["m00"]), int(moments["m01"] / moments["m00"])]
    else:
        center = [int((x1 + x2) / 2), int((y1 + y2) / 2)]

    return Hold(
        id=hold_id,
        polygon=poly_int.tolist(),
        bbox=[int(x1), int(y1), int(x2), int(y2)],
        center=center,
        area_px=int(cv2.countNonZero(mask)),
        confidence=round(det.confidence, 4),
        color=extract_color(image_bgr, mask),
    )


def run_pipeline(
    image_path: str | Path,
    model_path: str | Path | None = None,
    conf: float = DEFAULT_CONF,
    imgsz: int = DEFAULT_IMGSZ,
    device: str | None = None,
    detect_fn: DetectFn | None = None,
) -> tuple[DetectionOutput, np.ndarray]:
    """Full pipeline on one image. Returns (output, debug_image_bgr).

    `detect_fn` lets tests inject deterministic detections instead of the model.
    """
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"cannot read image: {image_path}")
    height, width = image_bgr.shape[:2]

    if detect_fn is None:
        resolved_model = Path(model_path or default_model_path())
        model_name = resolved_model.stem

        def detect_fn(p: Path) -> list[RawDetection]:
            return run_detection(p, resolved_model, conf=conf, imgsz=imgsz, device=device)
    else:
        model_name = "stub"

    holds = []
    for i, det in enumerate(detect_fn(image_path), start=1):
        mask = _polygon_to_mask(det.polygon, height, width)
        if cv2.countNonZero(mask) == 0:
            continue
        holds.append(_build_hold(i, det, image_bgr, mask))

    output = DetectionOutput(
        image=image_path.name, image_size=[width, height], model=model_name, holds=holds
    )
    return output, render_debug(image_bgr, holds)


def render_debug(image_bgr: np.ndarray, holds: list[Hold]) -> np.ndarray:
    """Outlines in each hold's detected color + 'id conf' labels."""
    canvas = image_bgr.copy()
    for hold in holds:
        bgr = DISPLAY_BGR[hold.color.name]
        pts = np.asarray(hold.polygon, dtype=np.int32)
        cv2.polylines(canvas, [pts], isClosed=True, color=bgr, thickness=2)
        label = f"{hold.id} {hold.confidence:.2f}"
        x, y = pts[pts[:, 1].argmin()]  # topmost vertex
        org = (int(x), max(12, int(y) - 5))
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, org, cv2.FONT_HERSHEY_SIMPLEX, 0.45, bgr, 1, cv2.LINE_AA)
    return canvas

"""Shared detection-evaluation helpers used by compare_baselines.py and vlm_baseline.py.

Everything here is CLASS-AGNOSTIC: a prediction is a box + confidence, a ground
truth is a box. We ask only "did the model draw a box where a hold is?" so that a
COCO model (classes person/chair/...) and a VLM (free-form) are judged on exactly
the same footing as the fine-tuned model. Metric = PASCAL-VOC all-point AP@IoU.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path, limit: int | None = None) -> list[Path]:
    imgs = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    return imgs[:limit] if limit else imgs


def load_gt_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    """Read a YOLO label file → (M, 4) xyxy boxes in pixels.

    Handles both segmentation polygons (class x1 y1 x2 y2 ...) and detection
    boxes (class cx cy bw bh). Coordinates in the file are normalised 0-1.
    """
    if not label_path.exists():
        return np.zeros((0, 4), dtype=np.float32)
    boxes: list[list[float]] = []
    for line in label_path.read_text().splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        coords = np.asarray(vals[1:], dtype=np.float32)
        if len(coords) == 4:  # cx cy bw bh
            cx, cy, bw, bh = coords
            xs = np.array([cx - bw / 2, cx + bw / 2])
            ys = np.array([cy - bh / 2, cy + bh / 2])
        else:  # polygon
            xs, ys = coords[0::2], coords[1::2]
        boxes.append([xs.min() * w, ys.min() * h, xs.max() * w, ys.max() * h])
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


def iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of one (4,) box against (M, 4) boxes → (M,)."""
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_m = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_m - inter
    return np.where(union > 0, inter / union, 0.0)


def _average_precision(confs: np.ndarray, tps: np.ndarray, n_gt: int) -> float:
    """VOC all-point AP from confidence-sorted TP flags."""
    if n_gt == 0 or len(confs) == 0:
        return 0.0
    order = np.argsort(-confs)
    tp = tps[order].astype(np.float64)
    fp = 1.0 - tp
    tp_cum, fp_cum = np.cumsum(tp), np.cumsum(fp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-12)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    for i in range(len(mpre) - 1, 0, -1):  # monotonic-decreasing envelope
        mpre[i - 1] = max(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def evaluate(
    predictions: dict[str, np.ndarray],
    ground_truth: dict[str, np.ndarray],
    iou_thr: float = 0.5,
    pr_conf: float = 0.35,
) -> dict[str, float]:
    """Class-agnostic detection metrics.

    predictions[image_id]  -> (N, 5) [x1, y1, x2, y2, conf]
    ground_truth[image_id] -> (M, 4) [x1, y1, x2, y2]

    Greedy IoU matching in descending-confidence order, one GT per prediction.
    Returns AP@iou_thr plus precision/recall/F1 at the pr_conf operating point.
    """
    records: list[tuple[float, str, np.ndarray]] = []
    total_gt = 0
    for img_id, gt in ground_truth.items():
        total_gt += len(gt)
        for row in predictions.get(img_id, np.zeros((0, 5))):
            records.append((float(row[4]), img_id, row[:4]))
    records.sort(key=lambda r: -r[0])

    matched: dict[str, set[int]] = {k: set() for k in ground_truth}
    confs = np.empty(len(records), dtype=np.float32)
    tps = np.zeros(len(records), dtype=np.float32)
    for i, (conf, img_id, box) in enumerate(records):
        confs[i] = conf
        gt = ground_truth[img_id]
        if len(gt) == 0:
            continue
        ious = iou_one_to_many(box, gt)
        taken = list(matched[img_id])
        if taken:
            ious[taken] = -1.0  # don't reuse a matched GT
        j = int(np.argmax(ious))
        if ious[j] >= iou_thr:
            tps[i] = 1.0
            matched[img_id].add(j)

    ap = _average_precision(confs, tps, total_gt)
    # records are conf-sorted desc, so {conf >= pr_conf} is a prefix
    n_keep = int(np.count_nonzero(confs >= pr_conf))
    tp_k = float(tps[:n_keep].sum())
    precision = tp_k / n_keep if n_keep else 0.0
    recall = tp_k / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return {
        "ap50": ap,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "n_pred": float(len(records)),
        "n_gt": float(total_gt),
        "pr_conf": pr_conf,
    }

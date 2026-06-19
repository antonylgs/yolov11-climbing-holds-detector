"""Compare the fine-tuned holds model against pretrained YOLO baselines on the
held-out test split, using one identical class-agnostic IoU scorer for all models.

Usage:
  uv run python scripts/compare_baselines.py \
      --test-dir data/dataset/test \
      --model trained=models/best.pt \
      --model coco=models/yolo11n-seg.pt \
      --out docs/eval --overlays 4

Writes <out>/comparison.csv, <out>/comparison.md, and (if --overlays N) N side-by-side
PNGs under <out>/overlays/. The point: the ONLY difference between rows is training,
so any mAP gap is attributable to fine-tuning, not to the metric.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from _eval import evaluate, list_images, load_gt_boxes

# Low conf at inference so the precision/recall curve is fully sampled (this is
# how `yolo val` computes mAP — NOT the deployment threshold).
EVAL_CONF = 0.001
COLORS = {"gt": (0, 200, 0), "trained": (255, 120, 0), "coco": (0, 0, 255)}  # BGR


def parse_models(pairs: list[str]) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pair in pairs:
        if "=" not in pair:
            raise SystemExit(f"--model expects name=path, got {pair!r}")
        name, path = pair.split("=", 1)
        out[name] = Path(path)
    return out


def predict_all(
    model_path: Path, images: list[Path], imgsz: int, device: str | None
) -> dict[str, np.ndarray]:
    """Run one model over every image → {stem: (N,5) [x1,y1,x2,y2,conf]}, class-agnostic."""
    from ultralytics import YOLO  # heavy import

    model = YOLO(str(model_path))
    preds: dict[str, np.ndarray] = {}
    for i, img in enumerate(images, 1):
        r = model.predict(str(img), conf=EVAL_CONF, imgsz=imgsz, device=device, verbose=False)[0]
        if r.boxes is None or len(r.boxes) == 0:
            preds[img.stem] = np.zeros((0, 5), dtype=np.float32)
        else:
            xyxy = r.boxes.xyxy.cpu().numpy()
            conf = r.boxes.conf.cpu().numpy().reshape(-1, 1)
            preds[img.stem] = np.hstack([xyxy, conf]).astype(np.float32)
        if i % 50 == 0 or i == len(images):
            print(f"  {model_path.name}: {i}/{len(images)} images", flush=True)
    return preds


def load_ground_truth(images: list[Path], labels_dir: Path) -> dict[str, np.ndarray]:
    gt: dict[str, np.ndarray] = {}
    for img in images:
        h, w = cv2.imread(str(img)).shape[:2]
        gt[img.stem] = load_gt_boxes(labels_dir / f"{img.stem}.txt", w, h)
    return gt


def draw_boxes(canvas: np.ndarray, boxes: np.ndarray, color, label: str) -> np.ndarray:
    out = canvas.copy()
    for b in boxes:
        x1, y1, x2, y2 = (int(v) for v in b[:4])
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(out, label, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2, cv2.LINE_AA)
    return out


def write_overlays(
    images: list[Path],
    gt: dict[str, np.ndarray],
    model_preds: dict[str, dict[str, np.ndarray]],
    out_dir: Path,
    n: int,
    vis_conf: float,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for img in images[:n]:
        base = cv2.imread(str(img))
        panels = [draw_boxes(base, gt[img.stem], COLORS["gt"], "ground truth")]
        for name, preds in model_preds.items():
            p = preds[img.stem]
            p = p[p[:, 4] >= vis_conf] if len(p) else p
            color = COLORS.get(name, (200, 200, 200))
            panels.append(draw_boxes(base, p, color, f"{name} ({len(p)})"))
        cv2.imwrite(str(out_dir / f"{img.stem}.png"), np.hstack(panels))
    print(f"  wrote {min(n, len(images))} overlays → {out_dir}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, default=Path("data/dataset/test"))
    p.add_argument("--model", action="append", dest="models", default=[], metavar="name=path")
    p.add_argument("--out", type=Path, default=Path("docs/eval"))
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--pr-conf", type=float, default=0.35, help="operating point for P/R/F1")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--overlays", type=int, default=0, help="write N side-by-side comparison PNGs")
    args = p.parse_args(argv)

    models = parse_models(args.models) or {
        "trained": Path("models/best.pt"),
        "coco": Path("models/yolo11n-seg.pt"),
    }
    images = list_images(args.test_dir / "images", args.limit)
    if not images:
        raise SystemExit(f"no images in {args.test_dir / 'images'}")
    print(f"{len(images)} test images, {len(models)} models, IoU={args.iou}")

    gt = load_ground_truth(images, args.test_dir / "labels")
    n_gt = sum(len(v) for v in gt.values())
    print(f"ground truth: {n_gt} holds")

    model_preds: dict[str, dict[str, np.ndarray]] = {}
    rows: list[dict] = []
    for name, path in models.items():
        if not path.exists():
            print(f"  ! skip {name}: {path} not found")
            continue
        print(f"running {name} ({path}) ...")
        preds = predict_all(path, images, args.imgsz, args.device)
        model_preds[name] = preds
        m = evaluate(preds, gt, iou_thr=args.iou, pr_conf=args.pr_conf)
        rows.append({"model": name, "weights": path.name, **m})

    args.out.mkdir(parents=True, exist_ok=True)
    fields = ["model", "weights", "ap50", "precision", "recall", "f1", "n_pred", "n_gt", "pr_conf"]
    with (args.out / "comparison.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    md = [
        f"# Baseline comparison (class-agnostic, IoU={args.iou}, "
        f"P/R @ conf≥{args.pr_conf})\n",
        f"Test split: {len(images)} images, {n_gt} ground-truth holds. "
        "Identical scorer for every row.\n",
        "| Model | Weights | mAP@50 | Precision | Recall | F1 |",
        "|---|---|---|---|---|---|",
    ]
    for r in rows:
        md.append(
            f"| {r['model']} | {r['weights']} | {r['ap50']:.3f} | "
            f"{r['precision']:.3f} | {r['recall']:.3f} | {r['f1']:.3f} |"
        )
    (args.out / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {args.out / 'comparison.csv'} and comparison.md")

    if args.overlays and model_preds:
        write_overlays(images, gt, model_preds, args.out / "overlays", args.overlays, args.pr_conf)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

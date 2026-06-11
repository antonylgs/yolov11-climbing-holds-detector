# PRD — Climbing Hold Detector

## 1. Summary

A pipeline that takes a photo of a climbing wall and outputs structured data for every hold: exact outline (segmentation mask), color, position, size, and confidence. Built by fine-tuning a pre-trained YOLO segmentation model on public climbing-hold datasets plus the owner's own labeled gym photos.

## 2. Goals / Non-goals

**Goals**
- Detect all visible climbing holds in a single wall photo (instance segmentation, single class `hold`)
- Per hold: polygon outline, bounding box, center, pixel area, dominant color (name + RGB), confidence
- CLI tool runnable on macOS (Apple Silicon, CPU/MPS) — no GPU required for inference
- Reproducible training pipeline (Colab notebook + config), so the model can be retrained as data grows
- JSON output + annotated debug image

**Non-goals (v1)**
- Real-time video / mobile deployment
- Hold type classification (jug, crimp, sloper…) — schema reserves a field, not implemented
- Route grouping (clustering holds into routes by color) — natural v2, out of scope
- Volume detection as separate class — labeled as `hold` in v1
- Web UI / API server

## 3. Users

- Primary: the project owner, learning fine-tuning; runs CLI locally
- Secondary: coding agents implementing issues from this PRD

## 4. Tech decisions (fixed — do not relitigate)

| Decision | Choice | Rationale |
|---|---|---|
| Task type | Instance segmentation | Holds are irregular; masks enable clean color extraction |
| Model | Ultralytics YOLO11s-seg (fallback yolov8s-seg) | Mature tooling, easy fine-tune, laptop-friendly |
| Classes | Single class `hold` | Color via post-processing, not learned classes |
| Color extraction | Classical CV on mask pixels (HSV, dominant cluster) | Accurate, no extra labels, tunable without retraining |
| Training compute | Google Colab GPU | Free/cheap; training script must also run locally on MPS |
| Inference | Local macOS, PyTorch (MPS/CPU) | Owner's laptop |
| Data format | YOLO segmentation format (txt polygons) | Native to Ultralytics |
| Dataset hosting/versioning | Roboflow project (free tier) | Annotation UI + dataset versioning + export in YOLO format |
| Language / tooling | Python 3.11+, `uv` for env, `ultralytics`, `opencv-python`, `numpy` | Standard stack |
| Image size (train+infer) | 1280 | Many small holds per wall photo; 640 default loses them |

## 5. Repository layout (target)

```
climbing-holds-detector/
├── LESSON.md / PRD.md / REQUIREMENTS.md
├── pyproject.toml
├── configs/
│   └── dataset.yaml            # YOLO data config (paths, class names)
├── data/                       # gitignored; raw + YOLO-format datasets
├── notebooks/
│   └── train_colab.ipynb       # Colab training notebook
├── src/holds_detector/
│   ├── detect.py               # inference: image -> masks
│   ├── color.py                # mask pixels -> color name + RGB
│   ├── pipeline.py             # detect + color -> JSON + debug image
│   ├── schema.py               # output dataclasses / JSON schema
│   └── cli.py                  # entrypoint
├── scripts/
│   ├── prepare_dataset.py      # merge public + own data, split, validate
│   ├── train.py                # local/Colab-shared training entry
│   └── eval.py                 # metrics + side-by-side prediction images
├── tests/
└── models/                     # gitignored; best.pt lives here
```

## 6. Output schema (contract)

```json
{
  "image": "wall.jpg",
  "image_size": [4032, 3024],
  "model": "yolo11s-seg-ft-v2",
  "holds": [
    {
      "id": 1,
      "polygon": [[x, y], ...],
      "bbox": [x1, y1, x2, y2],
      "center": [cx, cy],
      "area_px": 15234,
      "confidence": 0.91,
      "color": {
        "name": "red",
        "rgb": [201, 34, 52],
        "hsv": [350, 0.83, 0.79],
        "purity": 0.74
      },
      "hold_type": null
    }
  ]
}
```

- Coordinates in original image pixels (not resized).
- `color.name` ∈ {red, orange, yellow, green, blue, purple, pink, black, white, grey, wood, unknown}.
- `purity` = fraction of mask pixels belonging to the dominant color cluster (low ⇒ multi-colored or uncertain).
- `hold_type` always `null` in v1 (reserved).

## 7. Functional requirements

### F1 — Project scaffolding
`uv`-managed Python project, `pyproject.toml`, lint (ruff), pytest, `.gitignore` for `data/`, `models/`, runs. Smoke test: `ultralytics` imports and runs pretrained `yolo11n-seg` on a sample image.

### F2 — Dataset preparation
- Script to download/ingest Roboflow climbing-hold dataset export(s) (YOLO-seg format)
- Merge multiple sources into one dataset; remap all classes → single `hold`
- Deterministic 80/10/10 train/val/test split (split by *image*, seeded)
- Validation pass: flag empty labels, degenerate polygons, out-of-bounds coords
- Emit `configs/dataset.yaml`

### F3 — Training
- `scripts/train.py`: wraps `model.train()` with project defaults (imgsz=1280, epochs=100, single class), works on Colab CUDA and local MPS
- Colab notebook: mounts Drive, pulls dataset, trains, saves `best.pt` + metrics back to Drive
- Each run records: dataset version, base weights, key hyperparams (Ultralytics does most of this; make sure it's preserved)

### F4 — Evaluation
- `scripts/eval.py`: runs val on test split, prints mAP50 / mAP50-95 / precision / recall
- Renders side-by-side (ground truth vs prediction) images for the N worst test images — the human review artifact

### F5 — Inference + color pipeline
- `detect.py`: load `best.pt`, run on image at imgsz=1280, return masks scaled to original resolution, confidence threshold configurable (default 0.35)
- `color.py`: per mask → HSV conversion → k-means (k=3) on pixels → dominant cluster → hue→name mapping table (centralized, tunable) → `{name, rgb, hsv, purity}`
- Erode mask a few px before sampling (avoid edge/wall bleed); drop low-V shadow pixels
- `pipeline.py` + `cli.py`: `holds-detect photo.jpg -o out.json --debug-image out.png`
- Debug image: outlines colored by detected color name, id + confidence labels

### F6 — Model-assisted labeling support
- Script: run current model on a folder of unlabeled own-photos → write YOLO-seg label files → uploadable to Roboflow for human correction
- Doc snippet describing the labeling rules (label every hold incl. screw-on footholds; include volumes as `hold`; trace tight to hold edge; skip bolt holes/chalk marks)

### F7 — Tests
- Unit: color naming (synthetic masks of known colors, incl. dark/desaturated edge cases), schema serialization, polygon validation
- Integration: full pipeline on 2–3 checked-in small sample images with golden JSON (tolerance-based comparison)

## 8. Milestones

1. **M1 Baseline**: F1 + F2 (public data only) + F3 + F4 → first metrics + eyeball review on owner's gym photos
2. **M2 Pipeline**: F5 + F7 → end-to-end JSON from a photo using baseline model
3. **M3 Own data**: F6 → owner labels 50–150 gym photos → retrain → measurable improvement on a held-out own-gym test set
4. **M4 Polish**: threshold tuning, color table tuning on real outputs, README

## 9. Acceptance criteria

- `holds-detect <photo>` on a Mac produces valid JSON (schema above) + debug image in < 30 s
- mAP50 ≥ 0.80 on the held-out test split after M3
- Color name correct for ≥ 90% of detected holds on a 50-hold manually-checked sample (own gym photos, normal lighting)
- Re-running dataset prep + training with the same seed reproduces the dataset split exactly

## 10. Risks

| Risk | Mitigation |
|---|---|
| Public datasets too different from owner's gym | M3 exists precisely for this; budget labeling time |
| Small/distant holds missed | imgsz=1280; if insufficient, tile large images (SAHI) as v1.1 |
| Color confusion under colored gym lighting / faded holds | `purity` field + `unknown` fallback; tune hue table on real data |
| Colab free-tier limits | Small model + ~100 epochs fits sessions; checkpoint resume in notebook |

---
id: 003
title: Dataset preparation — merge sources, single class, seeded split, validation
labels: [ready-for-agent]
status: done
blocked_by: [001]
---

## What to build

`scripts/prepare_dataset.py`: takes one or more Roboflow YOLO-segmentation exports (directories), merges them into a single dataset under `data/`, remaps every class to the single class `hold`, performs a deterministic seeded 80/10/10 train/val/test split (split by image, never by annotation), and emits `configs/dataset.yaml` for Ultralytics.

A validation pass flags and reports: images with empty/missing label files, degenerate polygons (< 3 points, zero area), and out-of-bounds coordinates — with a `--strict` mode that fails on findings and a default mode that quarantines them.

Create a tiny fixture dataset (a handful of small images + hand-written YOLO-seg label files) checked into `tests/fixtures/` so the script is fully testable without downloading anything. Real public datasets are downloaded by the user (issue 005) — this script just consumes exports.

## Acceptance criteria

- [x] Running twice with the same seed produces byte-identical splits
- [x] Multiple source exports with different class lists merge into one single-class dataset
- [x] Validation pass catches each defect type (covered by tests using the fixture dataset)
- [x] Emitted `dataset.yaml` trains successfully with Ultralytics on the fixture dataset (1 epoch, smoke-level)
- [x] Tests pass via `uv run pytest`

## Blocked by

- 001

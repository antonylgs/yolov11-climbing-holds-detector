---
id: 001
title: Scaffold project + pretrained segmentation smoke test
labels: [ready-for-agent]
status: open
blocked_by: []
---

## What to build

Set up the Python project so a pretrained YOLO segmentation model runs end-to-end on a sample wall photo. This is the tracer bullet for the whole stack: environment → model download → inference → visible output.

`uv`-managed project per the PRD repo layout, with ruff + pytest wired up, and a `.gitignore` covering `data/`, `models/`, and Ultralytics run artifacts. Include a small sample wall image (royalty-free or synthetic) under `tests/fixtures/`. A smoke script (or test) loads pretrained `yolo11n-seg.pt`, runs it on the sample image, and writes an annotated output image — detections will be generic objects, not holds; that's expected and fine.

## Acceptance criteria

- [ ] `uv sync` from a clean clone installs everything; `ultralytics`, `opencv-python`, `numpy` importable
- [ ] `uv run pytest` passes, including a smoke test that runs pretrained yolo11n-seg on the fixture image (CPU/MPS, no GPU)
- [ ] Annotated output image is produced to a gitignored location
- [ ] `ruff check` passes; repo layout matches PRD §5

## Blocked by

None - can start immediately

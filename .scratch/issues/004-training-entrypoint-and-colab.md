---
id: 004
title: Training entrypoint + Colab notebook
labels: [ready-for-agent]
status: ready-for-human-verification
blocked_by: [003]
---

## What to build

`scripts/train.py` wrapping `model.train()` with project defaults: base weights `yolo11s-seg.pt`, imgsz 1280, epochs 100 (overridable), single class, auto device selection (CUDA on Colab, MPS locally, CPU fallback). Run artifacts (weights, metrics, args) land in a predictable gitignored location, and each run preserves dataset version + base weights + key hyperparams (Ultralytics records most of this — verify it survives).

`notebooks/train_colab.ipynb`: mounts Google Drive, pulls the prepared dataset from Drive, installs deps, calls the same `train.py` (no duplicated training logic), supports checkpoint resume across Colab session limits, and copies `best.pt` + result curves back to Drive at the end.

## Acceptance criteria

- [x] `uv run scripts/train.py --data tests/fixtures/.../dataset.yaml --epochs 2 --imgsz 320` completes locally on MPS/CPU and produces a `best.pt`
- [ ] Same script runs unmodified on Colab CUDA **(HUMAN: verified by owner during issue 005's training run)**
- [ ] Notebook is self-contained: fresh Colab session → trained weights on Drive, with resume-from-checkpoint cell **(HUMAN: same — agent verifies notebook structure only)**
- [x] Defaults match PRD §4 (yolo11s-seg, imgsz 1280, single class)

## Human touchpoint

Agents cannot execute Colab (needs owner's Google account). Complete all local criteria, then set `status: ready-for-human-verification` and leave the two HUMAN criteria unchecked — they get checked off as part of 005.

## Blocked by

- 003

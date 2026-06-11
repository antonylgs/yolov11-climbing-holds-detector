---
id: 006
title: Eval script — metrics + worst-case side-by-side renders
labels: [ready-for-agent]
status: open
blocked_by: [003, 004]
---

## What to build

`scripts/eval.py`: given weights and a dataset config, run validation on the **test** split and print mAP50, mAP50-95, precision, recall. Then render side-by-side images (ground truth outlines vs predicted outlines) for the N worst-scoring test images — the artifact a human reviews to decide what data to add next. Output to a gitignored eval directory with a small summary file (metrics + list of worst images).

Must work with the fixture dataset from 003 plus any 2-epoch fixture-trained weights from 004, so it's fully testable without the real baseline model.

## Acceptance criteria

- [ ] Prints the four PRD metrics for the test split (never train/val)
- [ ] `--worst N` produces N side-by-side ground-truth/prediction images
- [ ] Summary file includes metrics, weights path, dataset version
- [ ] Integration test runs end-to-end on fixture dataset + fixture weights

## Blocked by

- 003
- 004

---
id: 005
title: Baseline training run on public data (HUMAN)
labels: [needs-human]
status: open
blocked_by: [003, 004]
---

## What to build

Owner-executed milestone, not code. Using the tooling from 003/004:

1. Create Roboflow account; pick 1–2 public climbing-hold **segmentation** datasets from Roboflow Universe (polygon labels, not boxes); download as YOLO-seg exports
2. Run `prepare_dataset.py` on the exports; upload prepared dataset to Google Drive
3. Run the Colab notebook to train the baseline; ~1–3 h on free T4
4. Save `best.pt` to `models/` locally (gitignored) and note the run's final metrics in this issue
5. Eyeball check: run `holds-detect --model models/best.pt` on a few photos of YOUR gym; save outputs — these failure cases drive issues 007/008

## Acceptance criteria

- [ ] `models/best.pt` exists locally and on Drive
- [ ] Final val metrics (mAP50, mAP50-95, precision, recall) recorded below
- [ ] 3+ own-gym debug images saved with notes on what the baseline misses

## Blocked by

- 003
- 004

## Results

(fill in after the run)

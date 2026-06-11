---
id: 008
title: Photograph + label own gym, retrain, compare to baseline (HUMAN)
labels: [needs-human]
status: open
blocked_by: [005, 006, 007]
---

## What to build

Owner-executed milestone:

1. Take 50–150 photos of your gym: varied walls, angles, distances, lighting; include hard cases (faded holds, volumes, dense sections, dark corners); avoid blur
2. Pre-annotate with `scripts/preannotate.py` (007), upload to Roboflow, correct labels following `docs/labeling-rules.md`
3. Export, run `prepare_dataset.py` merging public + own data — keep a held-out own-gym test set
4. Retrain via Colab notebook
5. Run `eval.py` for old vs new weights on the own-gym test set; record both

## Acceptance criteria

- [ ] ≥ 50 own-gym images labeled and merged
- [ ] New `best.pt` trained on combined data
- [ ] Eval comparison recorded below: new model beats baseline on own-gym test set
- [ ] mAP50 ≥ 0.80 on held-out test split (PRD acceptance), or gap documented for 009

## Blocked by

- 005
- 006
- 007

## Results

(fill in: baseline vs retrained metrics)

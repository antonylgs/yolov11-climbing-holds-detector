---
id: 007
title: Model-assisted labeling — pre-annotate own photos + labeling rules doc
labels: [ready-for-agent]
status: open
blocked_by: [005]
---

## What to build

A script that runs the current best model over a folder of unlabeled gym photos and writes YOLO-seg label files alongside them, packaged so the owner can upload images + pre-annotations to Roboflow and only *correct* mistakes instead of drawing every polygon from scratch. Use a deliberately low confidence threshold (more pre-drawn candidates to delete beats holds to draw by hand).

Also write `docs/labeling-rules.md` — the consistency contract for all labeling: label every hold including screw-on footholds; volumes count as `hold`; trace tight to the hold edge; do NOT label bolt holes, chalk marks, or wall texture; one polygon per physically distinct hold even when touching.

## Acceptance criteria

- [ ] `scripts/preannotate.py <folder> --model <weights>` emits valid YOLO-seg labels for every image
- [ ] Output verified importable into a Roboflow annotation project (document the upload steps)
- [ ] Configurable confidence threshold, default lower than inference default
- [ ] `docs/labeling-rules.md` exists and covers the cases above
- [ ] Unit test: label-file format correctness on fixture image + fixture weights

## Blocked by

- 005 (needs a hold-trained model to produce useful pre-annotations)

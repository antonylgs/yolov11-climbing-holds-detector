---
id: 002
title: End-to-end CLI pipeline — photo → masks → color → JSON + debug image
labels: [ready-for-agent]
status: open
blocked_by: [001]
---

## What to build

The full inference pipeline as a CLI, using the *pretrained* model for now (detections will be poor on holds — irrelevant; the fine-tuned weights drop in later via a `--model` flag).

`holds-detect photo.jpg -o out.json --debug-image out.png` runs detection at imgsz 1280, scales masks back to original image resolution, then for each mask extracts the dominant color with classical CV: erode the mask a few pixels (avoid wall-edge bleed), drop low-value shadow pixels, k-means (k=3) in HSV, take the dominant cluster, map hue to a name via a single centralized, tunable table. Output JSON must match the PRD §6 contract exactly (polygon, bbox, center, area_px, confidence, color {name, rgb, hsv, purity}, hold_type: null). Debug image draws each outline in its detected color with id + confidence labels. Confidence threshold configurable, default 0.35.

Color names: red, orange, yellow, green, blue, purple, pink, black, white, grey, wood, unknown. Low purity or ambiguous hue ⇒ `unknown`, never a wrong guess.

## Acceptance criteria

- [ ] CLI produces schema-valid JSON + debug image from a photo in < 30 s on a Mac (CPU/MPS)
- [ ] Coordinates are in original image pixels regardless of inference resize
- [ ] Unit tests: color naming on synthetic masks of known colors, incl. dark/desaturated/multi-color edge cases; schema serialization
- [ ] Integration test: full pipeline on checked-in fixture image vs golden JSON (tolerance-based comparison)
- [ ] `--model` flag accepts any .pt weights path

## Blocked by

- 001

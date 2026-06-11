---
id: 009
title: Tune thresholds + color table on real outputs; README
labels: [ready-for-agent]
status: open
blocked_by: [008]
---

## What to build

Final polish pass using the retrained model and real gym photos from 008:

- Sweep the confidence threshold on the own-gym test set; pick the best precision/recall trade-off and make it the CLI default
- Audit color naming on a ~50-hold manually-checked sample from own-gym photos; adjust the centralized hue→name table and purity/`unknown` cutoffs until ≥ 90% correct (PRD acceptance); add any newly discovered edge cases to the color unit tests
- Write the README: what the project does, setup, the photo→JSON quickstart, retraining walkthrough, links to LESSON.md / PRD.md

## Acceptance criteria

- [ ] CLI default confidence threshold justified by a recorded sweep
- [ ] Color naming ≥ 90% correct on the 50-hold checked sample; sample + results committed
- [ ] New color edge cases covered by unit tests
- [ ] README enables a fresh user to go from clone → JSON output without reading other docs

## Blocked by

- 008

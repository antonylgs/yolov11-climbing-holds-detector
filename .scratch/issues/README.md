# Issue Tracker

Local issue tracker. One file per issue; frontmatter holds `status` (open/in-progress/done), `labels` (`ready-for-agent` = AFK-implementable, `needs-human` = owner must act), `blocked_by`.

Agents: pick the lowest-numbered `open` + `ready-for-agent` issue whose blockers are all `done`. Set `status: in-progress` while working, `done` when merged. Source spec: `../../PRD.md`.

## Board

| # | Issue | Labels | Blocked by |
|---|---|---|---|
| 001 | Scaffold + pretrained smoke test | ready-for-agent | — |
| 002 | End-to-end CLI pipeline (masks → color → JSON) | ready-for-agent | 001 |
| 003 | Dataset prep (merge, single class, split, validate) | ready-for-agent | 001 |
| 004 | Training entrypoint + Colab notebook | ready-for-agent | 003 |
| 005 | Baseline training run on public data | needs-human | 003, 004 |
| 006 | Eval script (metrics + worst-case renders) | ready-for-agent | 003, 004 |
| 007 | Model-assisted labeling + rules doc | ready-for-agent | 005 |
| 008 | Own gym photos, label, retrain | needs-human | 005, 006, 007 |
| 009 | Threshold/color tuning + README | ready-for-agent | 008 |

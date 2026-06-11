# What You'll Need

## Accounts (all free tiers OK)
- [ ] **Google account** — for Colab (training GPU) + Drive (store datasets/models)
- [ ] **Roboflow account** (roboflow.com) — download public climbing-hold datasets, annotate your own photos, version datasets
- [ ] **GitHub** — repo hosting + issues for the coding agents

## Software (local, macOS)
- [ ] Python 3.11+ (`brew install python` or via uv)
- [ ] `uv` (`brew install uv`) — Python env/package manager
- [ ] Packages (installed by the project): `ultralytics`, `opencv-python`, `numpy`, `pytest`, `ruff`
- [ ] Git

## Hardware
- [ ] Your Mac (Apple Silicon) — inference + dev. Nothing else needed locally.
- [ ] Colab GPU (T4 free tier) — training. ~1–3 h per training run for yolo11s-seg @ imgsz 1280.

## Data
- [ ] 1–2 public climbing-hold segmentation datasets from Roboflow Universe (search "climbing holds segmentation"; pick ones with polygon labels, not just boxes)
- [ ] **50–150 photos of your gym**, taken by you:
  - Varied: different walls, angles, distances, lighting (day/night)
  - Phone camera fine; avoid motion blur; shoot roughly perpendicular to wall
  - Include hard cases: faded holds, volumes, dense areas, dark corners
- [ ] A few hours of labeling time (model-assisted, so mostly corrections) — realistically 2–4 evenings

## Time budget (rough)
| Step | Time |
|---|---|
| Setup + baseline training on public data | 1 evening |
| Photographing your gym | 1 session at the gym |
| Labeling your photos | 2–4 evenings |
| Retrain + build pipeline + iterate | 2–3 evenings |

## Money
- $0 required. Optional: Colab Pro (~$10/mo) if free-tier GPU queues annoy you.

## Knowledge prep
- [ ] Read `LESSON.md` (you have it)
- [ ] Skim Ultralytics segmentation docs once: docs.ultralytics.com/tasks/segment

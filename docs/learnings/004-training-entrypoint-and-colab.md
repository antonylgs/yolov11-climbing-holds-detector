# Learning notes — Issue 004: Training entrypoint + Colab notebook

`configs/dataset.yaml` in → fine-tuned `best.pt` out. The same line of code runs on
the owner's Mac and on a Colab GPU, and a killed session loses nothing. This is the
issue where "reproducible training" (PRD §3) stops being a wish and becomes a button.

## What this issue accomplished

```
src/holds_detector/train.py    train() + device selection + run provenance (all the logic)
scripts/train.py               thin CLI: arg parsing only, no training logic
notebooks/train_colab.ipynb    Colab driver: mounts Drive, shells out to scripts/train.py
tests/test_train.py            6 tests, incl. a real 2-epoch fine-tune on the fixture
```

Try it (on the fixture from issue 003 — real data drops in the same way):
```bash
uv run python scripts/prepare_dataset.py \
    tests/fixtures/dataset/export_a tests/fixtures/dataset/export_b \
    --out /tmp/d/dataset --config /tmp/d/dataset.yaml
uv run python scripts/train.py --data /tmp/d/dataset.yaml \
    --weights models/yolo11n-seg.pt --epochs 2 --imgsz 320 --device cpu \
    --project /tmp/d/runs --name demo --no-export
cat /tmp/d/runs/demo/run_meta.json    # base weights + device + dataset seed
```
(Nano weights + `--no-export` keep the demo fast and leave `models/best.pt` alone.
The real run uses the PRD defaults: `yolo11s-seg`, imgsz 1280, 100 epochs.)

## The big ideas, one by one

### 1. One code path, two machines — the whole point of a training *script*
Training happens on Colab (free GPU) but must also run on the laptop (PRD §4). The
trap is two copies of the training call that drift apart — the notebook trains with
one set of args, the laptop with another, and "reproducible" quietly dies. The fix is
structural: *all* logic lives in `holds_detector.train.train()`; `scripts/train.py` is
pure argparse; the notebook does `!python scripts/train.py …`. The notebook contains
**zero** training logic, so there is nothing to keep in sync. Same pattern as issue
003 (`dataset.py` logic, thin `prepare_dataset.py` CLI) and 002 (`pipeline.py` logic,
thin `cli.py`) — it's the repo's spine: *logic in `src/`, entrypoints are skin.*

### 2. Device auto-selection: prefer the fast thing, fall back gracefully
`select_device()` is three lines of priority: explicit choice wins → else CUDA (Colab)
→ else Apple MPS (the laptop) → else CPU. The same unmodified command does the right
thing on a GPU box and on an M1. An explicit `--device cpu` always overrides — which is
exactly what the tests use, so they never depend on what hardware happens to be present.
`torch` is imported *inside* the function (deferred), same trick as `detect.py`'s YOLO
import: the heavy dependency only loads when you actually train, so `--help` stays instant.

### 3. Reproducibility you can *verify*, not just trust
"Each run records dataset version + base weights + hyperparams" (PRD §3) sounds like a
feature to build. Mostly it isn't — Ultralytics already writes `args.yaml` into every run
dir with the data path, base model, and every hyperparameter. The job here was to **verify
it survives** (a test asserts `data:` and `epochs:` are in `args.yaml`) rather than
reimplement it. The one thing Ultralytics *doesn't* put in one place is *which exact data*
this was: the dataset's split seed lives in issue 003's `validation_report.json`. So
`train.py` writes a tiny `run_meta.json` that pulls the seed in alongside the base weights
and resolved device. Now "what trained this model?" is a one-file answer — the hook that
makes "model v2 beats v1: same data, better training?" answerable (LESSON.md §6).

### 4. Resume, because Colab will kill you mid-run
Free Colab sessions die after a few hours; a 100-epoch run at imgsz 1280 won't always
finish in one sitting (PRD §10 risk). Two design choices make this survivable. First, runs
write *straight to Drive* (`--project <drive>/runs`), so checkpoints persist when the VM
vanishes. Second, `--resume` with no argument finds the most recently modified `last.pt`
and lets Ultralytics continue with that run's **stored** args. That last part matters:
resume deliberately *ignores* `--epochs`/`--imgsz`, so a resumed run can never silently
diverge from the one it's continuing. You can't fat-finger a resume into a different config.

### 5. The notebook is a driver, not a program
`train_colab.ipynb` only does the things that *must* happen in Colab and can't live in the
script: mount Drive, clone/pull the repo, `pip install -e .`, stage the dataset to local
disk, then call `scripts/train.py`. One non-obvious bit earns its keep: **stage the dataset
off Drive onto Colab's local disk** before training. Reading thousands of images over the
Drive FUSE mount per epoch is painfully slow; copying once and rewriting `dataset.yaml`'s
`path:` to the local copy turns IO from the bottleneck into a non-issue. Runs still go to
Drive so they persist — local for speed, Drive for durability.

### 6. `best.pt` → `models/best.pt`: closing the loop to inference
By default a finished run copies the winning weights to `models/best.pt` — exactly where
`detect.py:default_model_path()` looks first (issue 002). So "train" and "the CLI now uses
the new model" are one step locally. On Colab that's wrong (you want the weights on Drive,
not in an ephemeral VM's repo), so the notebook passes `--no-export` and you download
`best.pt` yourself. The default serves the common local case; the flag serves Colab.

## Review checklist

```bash
uv run pytest -q                 # 51 tests (6 new)
uv run ruff check .              # notebooks excluded (IPython magics aren't valid Python)
uv run python scripts/train.py --help
```
- Read `train.py` top to bottom — it's short; resume and `run_meta` carry the why.
- Skim the notebook's markdown cells: a fresh session reads top-to-bottom, with **5a
  (fresh)** vs **5b (resume)** as the one either/or choice.
- Confirm `select_device(None)` returns something sane for *your* machine (`mps` on the M1).

## Watch out for

- **Resuming an already-finished run** does nothing useful (Ultralytics sees 100/100 epochs
  done) and may spawn a stray default run dir. Resume is for *interrupted* runs.
- **`batch=16` at imgsz 1280** can OOM a small Colab GPU — the notebook calls this out; drop
  to 8. `--batch -1` (CUDA auto-batch) is available but CUDA-only.
- The local fixture run uses **nano** weights on purpose (fast, already cached). The real
  default is `yolo11s-seg`, which Ultralytics downloads on first use.

## Questions to test yourself

1. Why does the notebook shell out to `scripts/train.py` instead of importing and calling
   `train()` directly? What would break the day someone "improves" the notebook's copy?
2. Resume ignores `--epochs`. Why is that a feature and not a limitation?
3. What does `run_meta.json` add that Ultralytics' own `args.yaml` doesn't — and why is the
   *seed* specifically the thing worth pulling in?
4. Why stage the dataset to local disk on Colab when it's already on Drive?
5. Local runs export `best.pt` to `models/` by default but Colab passes `--no-export`. What
   goes wrong if you flip each default?

## What's next

Issue 005 (baseline training run) is where the owner downloads real Roboflow climbing-hold
exports, runs them through `prepare_dataset.py`, and drives *this* notebook on Colab for the
first real fine-tune — checking off the two HUMAN acceptance criteria left open here. Issue
006 (eval) then measures the `best.pt` this produces.

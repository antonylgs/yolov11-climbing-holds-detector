# Learning notes — Issue 002: End-to-end CLI pipeline

Photo → masks → color → JSON + debug image. The whole inference product, minus good weights.

## What this issue accomplished

```
src/holds_detector/
├── schema.py      the JSON contract (dataclasses mirroring PRD §6)
├── detect.py      YOLO inference → polygons in original-image pixels
├── color.py       mask pixels → color name (the classical CV part)
├── pipeline.py    glue: detect → per-hold color → schema + debug render
└── cli.py         `holds-detect photo.jpg -o out.json --debug-image out.png`
```

Try it:
```bash
uv run holds-detect tests/fixtures/sample_wall.jpg \
    -o data/outputs/wall.json --debug-image data/outputs/wall_debug.png --conf 0.05
open data/outputs/wall_debug.png
```
(`--conf 0.05` because pretrained weights barely fire on holds; the fine-tuned
model from issue 005 will work at the default 0.35 and just drops in via `--model`.)

## The big ideas, one by one

### 1. Separation of concerns = the module boundaries
Each file answers one question. `detect.py` knows *where* things are but nothing about color. `color.py` knows color but has never heard of YOLO — it takes "image + binary mask", which is why it's testable with synthetic 100×100 patches and no model. `schema.py` is pure data. This is LESSON.md §5 ("only make the neural network do what classical code can't") expressed as architecture.

### 2. Coordinates: the resize trap (acceptance criterion #2)
YOLO resizes to 1280 internally, but ultralytics' `masks.xy` already returns polygons scaled back to **original** image pixels — so `detect.py` does no math, just passes them through. The test `test_coordinates_within_original_image_bounds` pins this. If we ever switch libraries, that test is the tripwire.

### 3. Color extraction (`color.py`) — read it top to bottom, it's the lesson
Steps per mask, and *why*:
- **Erode 3 px** before sampling — mask edges straddle the hold/wall boundary; eroding keeps only confidently-inside pixels (wall bleed would grey out every color).
- **Drop V < 0.12 pixels** — shadows carry almost no hue information, only noise. (With a fallback: a genuinely *black* hold would lose all its pixels, so we only drop if enough survive.)
- **The hue-wrap problem**: hue is an angle — red lives at *both* 359° and 1°. Naive k-means on raw hue numbers would split red into two far-apart clusters. Fix: embed hue on a circle, `(s·cos h, s·sin h, v)`. This is the classic trick for clustering any circular quantity (angles, time-of-day, months).
- **k-means, k=3** then take the biggest cluster — separates the hold's true color from highlight + shadow regions.
- **Purity ≠ cluster share**: k-means happily splits even a perfectly uniform patch into 3 adjacent micro-clusters (~33% each). So purity counts all pixels within a *distance radius* of the dominant center instead. Uniform hold → ~1.0; half-red-half-blue → ~0.5 → below the 0.6 threshold → `unknown`. "Never a wrong guess" is implemented as this one threshold.
- **Naming order matters** (`name_from_hsv`): black (too dark) → white/grey (too desaturated) → wood (tan/brown band: orange-ish hue that's washed-out or dark) → hue table. Brown *is* dark orange — only V/S distinguish a wooden volume from an orange hold. All thresholds live in one `TUNING` dict / `HUE_TABLE` list — issue 009 tunes these against real photos without touching logic.

### 4. Dependency injection makes the integration test honest
Problem: a golden-JSON test through the *real* model breaks whenever ultralytics updates weights, and pretrained weights barely detect anything anyway. Solution: `run_pipeline(detect_fn=...)` accepts any "image → detections" callable. The integration test injects 4 fixed squares placed inside known fixture holds — everything downstream (mask rasterization, color, schema, rendering) runs for real and deterministically. The real model path is still covered end-to-end in `test_cli.py`, which asserts *schema validity and the <30 s budget*, not exact detections. Two tests, two different stability/coverage trade-offs.

### 5. Golden testing with tolerances
`tests/fixtures/golden_sample_wall.json` was generated once by running the stubbed pipeline, eyeballed (colors matched the planted blob colors exactly — e.g. red `[201,34,52]`), then checked in. The comparison allows ±3 px on coords, ±12 on RGB, ±0.15 on purity — JPEG decoding and OpenCV versions drift slightly across platforms; exact equality would make the test flaky for the wrong reasons.

### 6. Small things worth stealing
- `[project.scripts] holds-detect = "holds_detector.cli:main"` in pyproject → `uv sync` installs a real console command.
- `cli.main(argv)` takes an args list and returns an exit code → testable without subprocesses.
- JSON to stdout, status line to **stderr** — pipeable output stays clean.
- Debug labels drawn twice (thick black, then thin color) = readable text on any background.
- `unknown` renders magenta in the debug image: failure states should be loud, not invisible.

## Review checklist

```bash
uv run pytest -v          # 35 tests
uv run ruff check .
uv run holds-detect --help
```
- Read `color.py` fully — it's the densest learning payload in the repo.
- In `data/outputs/wall_debug.png`: check outline colors match what your eyes say.
- In `data/outputs/wall.json`: verify against PRD §6 by hand once.

## Questions to test yourself

1. Why does k-means on raw hue values break for red holds, and what's the fix?
2. Why is purity computed by distance-to-center over *all* pixels instead of "share of the dominant cluster"?
3. The integration test never loads YOLO. What exactly does it prove, and what does `test_cli.py` add on top?
4. A faded orange hold gets named `wood`. Which two `TUNING` entries would you adjust, and what's the risk?

## What's next

Issue 003 (dataset prep) — start feeding real climbing data toward training; the `--model` flag here is where its product (`best.pt`) eventually plugs in.

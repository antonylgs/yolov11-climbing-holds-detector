"""Prepare the training dataset from Roboflow YOLO-seg exports.

    uv run python scripts/prepare_dataset.py data/raw/export-a data/raw/export-b
    uv run python scripts/prepare_dataset.py data/raw/* --strict --seed 7

Merges the exports into data/dataset/ (single class: hold, seeded 80/10/10
split), quarantines defective pairs, and writes configs/dataset.yaml for
Ultralytics. All logic lives in holds_detector.dataset — this is just the
entrypoint."""

import sys

from holds_detector.dataset import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

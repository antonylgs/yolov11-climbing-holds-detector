#!/usr/bin/env python
"""Training entrypoint — shared by local runs and the Colab notebook.

    uv run python scripts/train.py --data configs/dataset.yaml          # full run, PRD defaults
    uv run python scripts/train.py --data configs/dataset.yaml --epochs 2 --imgsz 320  # smoke
    uv run python scripts/train.py --resume                             # continue newest run

All logic lives in ``holds_detector.train``; this file is just argument parsing so
the notebook and the laptop run identical code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from holds_detector.train import (  # noqa: E402
    DEFAULT_BASE_WEIGHTS,
    DEFAULT_BATCH,
    DEFAULT_EPOCHS,
    DEFAULT_IMGSZ,
    DEFAULT_NAME,
    DEFAULT_PROJECT,
    train,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("configs/dataset.yaml"),
        help="Ultralytics data yaml (from scripts/prepare_dataset.py)",
    )
    parser.add_argument("--weights", default=DEFAULT_BASE_WEIGHTS, help="base weights to fine-tune")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ)
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="-1 for CUDA auto-batch")
    parser.add_argument("--device", default=None, help="cpu|mps|cuda|0; auto-selected if unset")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help="run output dir (gitignored)")
    parser.add_argument("--name", default=DEFAULT_NAME, help="run name under --project")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="resume newest run, or a specific weights/last.pt path",
    )
    parser.add_argument(
        "--no-export",
        action="store_true",
        help="don't copy best.pt to models/ (keep an existing inference model)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.resume and not args.data.exists():
        print(f"error: data config not found: {args.data}", file=sys.stderr)
        print("run scripts/prepare_dataset.py first, or pass --data", file=sys.stderr)
        return 1
    try:
        train(
            data=args.data,
            base_weights=args.weights,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            project=args.project,
            name=args.name,
            resume=args.resume,
            export_best=not args.no_export,
        )
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

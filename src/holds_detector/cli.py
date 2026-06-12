"""CLI entrypoint: holds-detect photo.jpg -o out.json --debug-image out.png"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

from holds_detector.detect import DEFAULT_CONF, DEFAULT_IMGSZ
from holds_detector.pipeline import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="holds-detect",
        description="Detect climbing holds in a wall photo → JSON + debug image",
    )
    parser.add_argument("image", type=Path, help="input wall photo")
    parser.add_argument("-o", "--output", type=Path, help="output JSON path (default: stdout)")
    parser.add_argument("--debug-image", type=Path, help="write annotated image here")
    parser.add_argument(
        "--model", type=Path, default=None, help="weights .pt path (default: models/best.pt "
        "if it exists, else pretrained yolo11n-seg)"
    )
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF, help="confidence threshold")
    parser.add_argument("--imgsz", type=int, default=DEFAULT_IMGSZ, help="inference image size")
    parser.add_argument(
        "--device", default=None, help="torch device (cpu, mps, cuda); auto if unset"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.image.exists():
        print(f"error: image not found: {args.image}", file=sys.stderr)
        return 1

    start = time.perf_counter()
    output, debug_image = run_pipeline(
        args.image, model_path=args.model, conf=args.conf, imgsz=args.imgsz, device=args.device
    )
    elapsed = time.perf_counter() - start

    json_text = output.to_json()
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text + "\n")
    else:
        print(json_text)

    if args.debug_image:
        args.debug_image.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(args.debug_image), debug_image)

    print(
        f"{len(output.holds)} holds in {elapsed:.1f}s"
        + (f" → {args.output}" if args.output else "")
        + (f", debug → {args.debug_image}" if args.debug_image else ""),
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

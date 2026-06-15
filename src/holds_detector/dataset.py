"""Dataset preparation: merge Roboflow YOLO-seg exports into one training-ready dataset.

Input: one or more export directories in Roboflow's YOLO layout — any number of
split subdirectories (train/valid/test or none), each holding ``images/`` and
``labels/`` with one ``.txt`` per image. Label line format: ``class x1 y1 x2 y2 ...``
with coordinates normalized to [0, 1].

Output (everything regenerated from scratch on each run):

    <out>/
    ├── train/{images,labels}/      80% of valid images (seeded, deterministic)
    ├── val/{images,labels}/        10%
    ├── test/{images,labels}/       10%
    ├── quarantine/                 defective image+label pairs, kept for inspection
    └── validation_report.json
    <config>                        Ultralytics data yaml (single class: hold)

Every class id from every source is remapped to 0 (``hold``) — the model never
learns color or hold type; see LESSON.md §5. The split is by image, never by
annotation, so no image's labels leak across splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path

CLASS_NAME = "hold"
DEFAULT_SEED = 42
DEFAULT_RATIOS = (0.8, 0.1, 0.1)
SPLITS = ("train", "val", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Filesystems cap a single path component at 255 bytes. Roboflow stems are already
# huge (original filename + ".rf." + hash), and we prepend source/split context, so
# we cap the output stem well under the limit — leaving room for the ".jpg"/".txt"
# suffix — and preserve uniqueness with a short hash of the full original name.
_MAX_STEM_LEN = 120

# Normalized shoelace area below this is "zero": a 4×4 px polygon in a 4000 px
# photo is still ~5e-7, so 1e-12 only catches genuinely collinear/duplicate points.
_AREA_EPS = 1e-12


@dataclass(frozen=True)
class Finding:
    """One validation defect on an image/label pair."""

    # missing_label | empty_label | malformed_line | too_few_points | zero_area | out_of_bounds
    kind: str
    detail: str


@dataclass
class Pair:
    """One image with its (possibly missing) label file and unique output stem."""

    image: Path
    label: Path
    name: str
    findings: list[Finding] = field(default_factory=list)


def discover_pairs(source: Path, source_id: str) -> list[Pair]:
    """Find every image under any ``images/`` directory in the export, paired with
    its sibling ``labels/<stem>.txt``. Output names embed the source id and the
    path inside the export, so merged sources can never collide."""
    images = sorted(
        p
        for p in source.rglob("*")
        if p.suffix.lower() in IMAGE_SUFFIXES and p.parent.name == "images"
    )
    pairs: list[Pair] = []
    for image in images:
        label = image.parent.parent / "labels" / f"{image.stem}.txt"
        rel_parts = [part for part in image.relative_to(source).parts[:-1] if part != "images"]
        name = _safe_stem("_".join([source_id, *rel_parts, image.stem]))
        pairs.append(Pair(image=image, label=label, name=name))
    return pairs


def _safe_stem(raw: str, max_len: int = _MAX_STEM_LEN) -> str:
    """Bound an output stem to the filesystem limit without losing uniqueness.

    Short names pass through unchanged. Long ones (Roboflow's monster filenames) are
    truncated and tagged with an 8-char hash of the *full* original name — so two
    stems that share a truncated prefix still get distinct outputs, deterministically."""
    if len(raw) <= max_len:
        return raw
    digest = hashlib.sha1(raw.encode()).hexdigest()[:8]
    return f"{raw[: max_len - 9]}_{digest}"


def shoelace_area(points: list[tuple[float, float]]) -> float:
    """Polygon area in normalized units (always >= 0)."""
    area = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1], strict=True):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def validate_pair(pair: Pair) -> list[Finding]:
    """Check one pair for every defect type. Any finding quarantines the whole pair."""
    if not pair.label.exists():
        return [Finding("missing_label", f"expected {pair.label.name}")]
    lines = [line for line in pair.label.read_text().splitlines() if line.strip()]
    if not lines:
        return [Finding("empty_label", "label file has no annotations")]

    findings: list[Finding] = []
    for lineno, line in enumerate(lines, start=1):
        tokens = line.split()
        try:
            values = [float(token) for token in tokens]
        except ValueError:
            findings.append(Finding("malformed_line", f"line {lineno}: non-numeric token"))
            continue
        coords = values[1:]
        if len(coords) % 2 != 0:
            findings.append(Finding("malformed_line", f"line {lineno}: odd coordinate count"))
            continue
        points = list(zip(coords[::2], coords[1::2], strict=True))
        if len(points) < 3:
            findings.append(Finding("too_few_points", f"line {lineno}: {len(points)} points"))
            continue
        if any(not 0.0 <= c <= 1.0 for c in coords):
            findings.append(Finding("out_of_bounds", f"line {lineno}: coordinate outside [0, 1]"))
        if shoelace_area(points) <= _AREA_EPS:
            findings.append(Finding("zero_area", f"line {lineno}: degenerate polygon"))
    return findings


def split_pairs(
    pairs: list[Pair], seed: int, ratios: tuple[float, float, float]
) -> dict[str, list[Pair]]:
    """Deterministic seeded split by image. Sorting by unique name before the
    seeded shuffle makes the result independent of filesystem listing order —
    same inputs + same seed = byte-identical output, always."""
    ordered = sorted(pairs, key=lambda p: p.name)
    random.Random(seed).shuffle(ordered)
    n = len(ordered)
    # val/test get at least one image each once there's anything to spare
    n_val = max(1, int(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, int(n * ratios[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def _write_remapped_label(src: Path, dst: Path) -> None:
    """Copy a label file with every class id replaced by 0. Coordinate tokens are
    passed through verbatim — no float round-trip, no precision drift."""
    out_lines = []
    for line in src.read_text().splitlines():
        if not line.strip():
            continue
        tokens = line.split()
        out_lines.append(" ".join(["0", *tokens[1:]]))
    dst.write_text("\n".join(out_lines) + "\n")


def _write_config(config_path: Path, out_dir: Path) -> None:
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(
        "# generated by scripts/prepare_dataset.py — do not edit, rerun the script\n"
        f"path: {out_dir}\n"
        "train: train/images\n"
        "val: val/images\n"
        "test: test/images\n"
        "nc: 1\n"
        "names:\n"
        f"  0: {CLASS_NAME}\n"
    )


def prepare(
    sources: list[Path],
    out_dir: Path,
    config_path: Path,
    seed: int = DEFAULT_SEED,
    ratios: tuple[float, float, float] = DEFAULT_RATIOS,
    strict: bool = False,
) -> dict:
    """Merge, validate, split, and write the dataset. Returns the report dict;
    ``report["ok"]`` is False only in strict mode with findings (nothing written)."""
    if abs(sum(ratios) - 1.0) > 1e-9:
        raise ValueError(f"split ratios must sum to 1, got {ratios}")
    pairs: list[Pair] = []
    for index, source in enumerate(sources):
        source = source.resolve()
        if not source.is_dir():
            raise FileNotFoundError(f"source export not found: {source}")
        found = discover_pairs(source, f"{index:02d}_{source.name}")
        if not found:
            raise FileNotFoundError(f"no images under any images/ directory in: {source}")
        pairs.extend(found)

    for pair in pairs:
        pair.findings = validate_pair(pair)
    good = [p for p in pairs if not p.findings]
    bad = [p for p in pairs if p.findings]

    report = {
        "ok": not (strict and bad),
        "images_found": len(pairs),
        "images_valid": len(good),
        "quarantined": [
            {"name": p.name, "findings": [{"kind": f.kind, "detail": f.detail} for f in p.findings]}
            for p in sorted(bad, key=lambda p: p.name)
        ],
        "splits": {},
        "seed": seed,
    }
    if strict and bad:
        return report

    out_dir = out_dir.resolve()
    if out_dir.exists():
        shutil.rmtree(out_dir)

    quarantine_dir = out_dir / "quarantine"
    for pair in bad:
        quarantine_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(pair.image, quarantine_dir / f"{pair.name}{pair.image.suffix.lower()}")
        if pair.label.exists():
            shutil.copyfile(pair.label, quarantine_dir / f"{pair.name}.txt")

    splits = split_pairs(good, seed, ratios)
    for split_name, split_pairs_ in splits.items():
        images_dir = out_dir / split_name / "images"
        labels_dir = out_dir / split_name / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)
        for pair in split_pairs_:
            shutil.copyfile(pair.image, images_dir / f"{pair.name}{pair.image.suffix.lower()}")
            _write_remapped_label(pair.label, labels_dir / f"{pair.name}.txt")
        report["splits"][split_name] = len(split_pairs_)

    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2) + "\n")
    _write_config(config_path.resolve(), out_dir)
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Merge Roboflow YOLO-seg exports into one single-class dataset "
        "with a deterministic train/val/test split."
    )
    parser.add_argument("sources", nargs="+", type=Path, help="Roboflow export directories")
    parser.add_argument("--out", type=Path, default=Path("data/dataset"))
    parser.add_argument("--config", type=Path, default=Path("configs/dataset.yaml"))
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--ratios",
        nargs=3,
        type=float,
        default=list(DEFAULT_RATIOS),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="fail on any validation finding instead of quarantining",
    )
    args = parser.parse_args(argv)

    try:
        report = prepare(
            sources=args.sources,
            out_dir=args.out,
            config_path=args.config,
            seed=args.seed,
            ratios=tuple(args.ratios),
            strict=args.strict,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    for entry in report["quarantined"]:
        kinds = ", ".join(f["kind"] for f in entry["findings"])
        print(f"quarantined {entry['name']}: {kinds}", file=sys.stderr)
    if not report["ok"]:
        print(f"strict mode: {len(report['quarantined'])} defective image(s)", file=sys.stderr)
        return 1
    counts = " ".join(f"{name}={count}" for name, count in report["splits"].items())
    print(
        f"dataset ready: {report['images_valid']}/{report['images_found']} images kept, "
        f"{counts} (seed {report['seed']}) -> {args.config}",
        file=sys.stderr,
    )
    return 0

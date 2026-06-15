"""Generate the Roboflow-export-shaped fixture datasets under tests/fixtures/dataset/.

Three fake exports, so prepare_dataset.py is fully testable without downloads:

- export_a: two splits (train/valid), two classes (hold, volume) — exercises
  split discovery and class remapping
- export_b: one split, one class with a different name (climbing-hold) —
  exercises merging sources with mismatched class lists
- export_defects: one image per validation defect type

Images are tiny (96x96) synthetic walls with colored blobs; labels are written
from the exact polygons drawn, so the fixture is also good enough for a 1-epoch
training smoke test. Deterministic (seeded). Run with:

    uv run python scripts/generate_dataset_fixture.py
"""

from pathlib import Path

import cv2
import numpy as np

SIZE = 96
SEED = 7
FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "dataset"

# BGR, matching common hold colors (same palette idea as generate_fixture.py)
HOLD_COLORS = [(52, 34, 201), (190, 90, 30), (70, 160, 40), (40, 200, 230), (160, 50, 130)]


def make_wall(rng: np.random.Generator) -> np.ndarray:
    wall = np.full((SIZE, SIZE, 3), 165, dtype=np.uint8)
    noise = rng.normal(0, 8, size=(SIZE, SIZE, 1)).astype(np.int16)
    return np.clip(wall.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def random_hold_polygon(rng: np.random.Generator) -> np.ndarray:
    """Irregular blob polygon, fully inside the image. Returns (N, 2) pixel coords."""
    radius = int(rng.integers(10, 22))
    cx = int(rng.integers(radius + 2, SIZE - radius - 2))
    cy = int(rng.integers(radius + 2, SIZE - radius - 2))
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=int(rng.integers(6, 10))))
    radii = rng.uniform(0.6, 1.0, size=angles.shape) * radius
    pts = np.stack([cx + radii * np.cos(angles), cy + radii * np.sin(angles)], axis=1)
    return np.clip(pts, 1, SIZE - 2)


def label_line(class_id: int, polygon: np.ndarray) -> str:
    coords = " ".join(f"{value / SIZE:.6f}" for point in polygon for value in point)
    return f"{class_id} {coords}"


def write_pair(
    split_dir: Path, stem: str, rng: np.random.Generator, n_holds: int, class_ids: list[int]
) -> None:
    """One image with n_holds drawn blobs + the matching label file."""
    wall = make_wall(rng)
    lines = []
    for i in range(n_holds):
        polygon = random_hold_polygon(rng)
        color = HOLD_COLORS[int(rng.integers(len(HOLD_COLORS)))]
        cv2.fillPoly(wall, [polygon.astype(np.int32)], color)
        lines.append(label_line(class_ids[i % len(class_ids)], polygon))
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(split_dir / "images" / f"{stem}.jpg"), wall, [cv2.IMWRITE_JPEG_QUALITY, 90])
    (split_dir / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n")


def write_image_only(split_dir: Path, stem: str, rng: np.random.Generator) -> Path:
    (split_dir / "images").mkdir(parents=True, exist_ok=True)
    (split_dir / "labels").mkdir(parents=True, exist_ok=True)
    cv2.imwrite(
        str(split_dir / "images" / f"{stem}.jpg"), make_wall(rng), [cv2.IMWRITE_JPEG_QUALITY, 90]
    )
    return split_dir / "labels" / f"{stem}.txt"


def main() -> None:
    rng = np.random.default_rng(SEED)

    # export_a: Roboflow-style train/valid splits, classes [hold, volume]
    export_a = FIXTURE_ROOT / "export_a"
    (export_a / "data.yaml").parent.mkdir(parents=True, exist_ok=True)
    (export_a / "data.yaml").write_text("nc: 2\nnames: ['hold', 'volume']\n")
    for i in range(4):
        write_pair(export_a / "train", f"wall_{i}", rng, n_holds=3, class_ids=[0, 1])
    for i in range(2):
        write_pair(export_a / "valid", f"wall_{i}", rng, n_holds=2, class_ids=[0, 1])

    # export_b: flat single split, one class with a different name
    export_b = FIXTURE_ROOT / "export_b"
    export_b.mkdir(parents=True, exist_ok=True)
    (export_b / "data.yaml").write_text("nc: 1\nnames: ['climbing-hold']\n")
    for i in range(4):
        write_pair(export_b / "train", f"gym_{i}", rng, n_holds=2, class_ids=[0])

    # export_defects: one pair per defect type (labels hand-written below)
    defects = FIXTURE_ROOT / "export_defects" / "train"
    write_image_only(defects, "missing_label", rng)  # no .txt at all
    write_image_only(defects, "empty_label", rng).write_text("")
    write_image_only(defects, "two_points", rng).write_text("0 0.10 0.10 0.50 0.50\n")
    write_image_only(defects, "zero_area", rng).write_text(
        "0 0.10 0.10 0.50 0.50 0.90 0.90\n"  # collinear -> zero area
    )
    write_image_only(defects, "out_of_bounds", rng).write_text(
        "0 0.20 0.20 1.20 0.20 0.70 0.80\n"  # x=1.2 outside [0,1]
    )

    print(f"Wrote fixture exports under {FIXTURE_ROOT}")


if __name__ == "__main__":
    main()

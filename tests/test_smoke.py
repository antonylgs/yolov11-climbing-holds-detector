"""Smoke test: the full stack works end to end.

Loads the pretrained yolo11n-seg model (downloaded to models/ on first run),
runs inference on the sample wall fixture, and writes an annotated image to a
gitignored location. The pretrained model knows COCO objects, not climbing
holds, so we assert the *machinery* works (model loads, inference runs, output
image written) — not that any holds are detected.
"""

from pathlib import Path

import cv2
import pytest
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_wall.jpg"
MODEL_PATH = ROOT / "models" / "yolo11n-seg.pt"
OUTPUT_DIR = ROOT / "data" / "outputs"


def test_fixture_image_is_valid() -> None:
    img = cv2.imread(str(FIXTURE))
    assert img is not None, f"fixture missing or unreadable: {FIXTURE}"
    height, width = img.shape[:2]
    assert width >= 640 and height >= 480


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_pretrained_seg_inference_smoke() -> None:
    # YOLO() downloads the named checkpoint to this path if it doesn't exist yet
    model = YOLO(str(MODEL_PATH))

    results = model.predict(str(FIXTURE), imgsz=640, device="cpu", verbose=False)
    assert len(results) == 1

    result = results[0]
    # Pretrained COCO classes won't match holds — zero detections is acceptable.
    # What matters: inference ran and produced a well-formed Results object.
    assert result.boxes is not None

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "smoke_annotated.jpg"
    result.save(filename=str(out_path))
    assert out_path.exists() and out_path.stat().st_size > 0

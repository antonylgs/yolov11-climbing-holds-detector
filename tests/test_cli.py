"""End-to-end CLI with the real (pretrained) model.

Pretrained weights don't know holds, so hold *count* is not asserted — only
that the CLI runs within budget and emits schema-valid JSON + a debug image.
"""

import json
import time
from pathlib import Path

from holds_detector.cli import main

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_wall.jpg"


def test_cli_end_to_end(tmp_path):
    out_json = tmp_path / "out.json"
    out_img = tmp_path / "debug.png"

    start = time.perf_counter()
    code = main([str(FIXTURE), "-o", str(out_json), "--debug-image", str(out_img)])
    elapsed = time.perf_counter() - start

    assert code == 0
    assert elapsed < 30, f"pipeline took {elapsed:.1f}s, budget is 30s"
    assert out_img.exists() and out_img.stat().st_size > 0

    data = json.loads(out_json.read_text())
    assert set(data) == {"image", "image_size", "model", "holds"}
    assert data["image"] == FIXTURE.name
    assert data["image_size"] == [960, 720]
    for hold in data["holds"]:  # may be empty with pretrained weights — fine
        assert set(hold) == {
            "id", "polygon", "bbox", "center", "area_px", "confidence", "color", "hold_type",
        }  # fmt: skip


def test_cli_missing_image_fails_cleanly(tmp_path):
    assert main([str(tmp_path / "nope.jpg")]) == 1


def test_cli_model_flag_accepts_path(tmp_path):
    """--model takes any .pt path (the fine-tuned weights drop-in)."""
    out_json = tmp_path / "out.json"
    model = ROOT / "models" / "yolo11n-seg.pt"
    code = main([str(FIXTURE), "-o", str(out_json), "--model", str(model), "--imgsz", "640"])
    assert code == 0
    assert json.loads(out_json.read_text())["model"] == "yolo11n-seg"

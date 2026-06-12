"""Integration: full pipeline on the fixture image vs golden JSON.

Detection is stubbed with fixed polygons placed inside known holds of the
synthetic wall, so the masks→color→schema path is fully deterministic and the
golden comparison is meaningful regardless of model weights. The real-model
path is covered by test_cli.py.
"""

import json
from pathlib import Path

import numpy as np

from holds_detector.detect import RawDetection
from holds_detector.pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "sample_wall.jpg"
GOLDEN = ROOT / "tests" / "fixtures" / "golden_sample_wall.json"

# Squares centered inside four holds of the fixture (blue, orange, purple, red)
STUB_HOLDS = [
    ((236, 112), 14, 0.95),
    ((57, 601), 14, 0.88),
    ((622, 456), 14, 0.77),
    ((775, 586), 12, 0.66),
]


def square(center: tuple[int, int], half: int) -> np.ndarray:
    cx, cy = center
    return np.array(
        [
            [cx - half, cy - half],
            [cx + half, cy - half],
            [cx + half, cy + half],
            [cx - half, cy + half],
        ],
        dtype=np.float64,
    )


def stub_detect(_path: Path) -> list[RawDetection]:
    return [
        RawDetection(polygon=square(c, h), confidence=conf) for c, h, conf in STUB_HOLDS
    ]


def run() -> tuple[dict, np.ndarray]:
    output, debug = run_pipeline(FIXTURE, detect_fn=stub_detect)
    return output.to_dict(), debug


def test_pipeline_matches_golden():
    actual, _ = run()
    golden = json.loads(GOLDEN.read_text())

    assert actual["image"] == golden["image"]
    assert actual["image_size"] == golden["image_size"]
    assert len(actual["holds"]) == len(golden["holds"])

    for got, want in zip(actual["holds"], golden["holds"], strict=True):
        assert got["id"] == want["id"]
        assert got["color"]["name"] == want["color"]["name"]
        assert got["hold_type"] is None
        # tolerance-based numeric comparison (jpeg/opencv versions may drift)
        assert np.allclose(got["bbox"], want["bbox"], atol=3)
        assert np.allclose(got["center"], want["center"], atol=3)
        assert abs(got["area_px"] - want["area_px"]) <= 0.1 * want["area_px"]
        assert got["confidence"] == want["confidence"]
        assert np.allclose(got["color"]["rgb"], want["color"]["rgb"], atol=12)
        assert abs(got["color"]["purity"] - want["color"]["purity"]) <= 0.15


def test_debug_image_shape_and_drawing():
    actual, debug = run()
    h, w = debug.shape[:2]
    assert [w, h] == actual["image_size"]
    # drawing happened: debug differs from the input somewhere
    import cv2

    original = cv2.imread(str(FIXTURE))
    assert (debug != original).any()


def test_coordinates_within_original_image_bounds():
    actual, _ = run()
    width, height = actual["image_size"]
    for hold in actual["holds"]:
        xs = [p[0] for p in hold["polygon"]]
        ys = [p[1] for p in hold["polygon"]]
        assert 0 <= min(xs) and max(xs) < width
        assert 0 <= min(ys) and max(ys) < height

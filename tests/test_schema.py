"""Schema serialization matches the PRD §6 contract."""

import json

from holds_detector.schema import COLOR_NAMES, ColorInfo, DetectionOutput, Hold


def make_output() -> DetectionOutput:
    hold = Hold(
        id=1,
        polygon=[[10, 10], [50, 12], [48, 55], [12, 50]],
        bbox=[10, 10, 50, 55],
        center=[30, 31],
        area_px=1520,
        confidence=0.91,
        color=ColorInfo(name="red", rgb=[201, 34, 52], hsv=[350.0, 0.83, 0.79], purity=0.74),
    )
    return DetectionOutput(
        image="wall.jpg", image_size=[4032, 3024], model="yolo11s-seg-ft-v2", holds=[hold]
    )


def test_json_roundtrip_matches_contract():
    data = json.loads(make_output().to_json())

    assert set(data) == {"image", "image_size", "model", "holds"}
    assert data["image_size"] == [4032, 3024]

    hold = data["holds"][0]
    assert set(hold) == {
        "id", "polygon", "bbox", "center", "area_px", "confidence", "color", "hold_type",
    }  # fmt: skip
    assert hold["hold_type"] is None  # serialized as JSON null, reserved for v2
    assert hold["polygon"] == [[10, 10], [50, 12], [48, 55], [12, 50]]
    assert set(hold["color"]) == {"name", "rgb", "hsv", "purity"}
    assert hold["color"]["name"] in COLOR_NAMES


def test_empty_output_is_valid():
    out = DetectionOutput(image="x.jpg", image_size=[100, 100], model="m")
    assert json.loads(out.to_json())["holds"] == []

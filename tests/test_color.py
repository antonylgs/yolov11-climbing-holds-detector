"""Color naming on synthetic masks of known colors, incl. edge cases."""

import numpy as np
import pytest

from holds_detector.color import extract_color, name_from_hsv

SIZE = 100


def solid_patch(bgr: tuple[int, int, int], noise: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A SIZE×SIZE image of one color (optional gaussian noise) + full mask."""
    img = np.full((SIZE, SIZE, 3), bgr, dtype=np.uint8)
    if noise:
        rng = np.random.default_rng(7)
        img = np.clip(img.astype(np.int16) + rng.normal(0, noise, img.shape), 0, 255).astype(
            np.uint8
        )
    return img, np.ones((SIZE, SIZE), dtype=np.uint8)


@pytest.mark.parametrize(
    ("bgr", "expected"),
    [
        ((36, 28, 221), "red"),
        ((30, 140, 245), "orange"),
        ((40, 210, 235), "yellow"),
        ((60, 170, 50), "green"),
        ((200, 90, 25), "blue"),
        ((180, 40, 120), "purple"),
        ((180, 105, 255), "pink"),
        ((30, 30, 30), "black"),
        ((245, 245, 245), "white"),
        ((128, 128, 128), "grey"),
        ((95, 140, 180), "wood"),  # tan
        ((40, 75, 110), "wood"),  # dark brown
    ],
)
def test_solid_colors(bgr, expected):
    info = extract_color(*solid_patch(bgr))
    assert info.name == expected
    assert info.purity > 0.9  # uniform patch → near-total dominance


def test_noisy_color_still_named():
    info = extract_color(*solid_patch((36, 28, 221), noise=12))
    assert info.name == "red"
    assert info.purity > 0.8


def test_dark_desaturated_is_grey():
    info = extract_color(*solid_patch((65, 60, 60)))
    assert info.name == "grey"


def test_multicolor_mask_is_unknown_not_a_guess():
    img = np.zeros((SIZE, SIZE, 3), dtype=np.uint8)
    img[:, : SIZE // 2] = (36, 28, 221)  # red half
    img[:, SIZE // 2 :] = (200, 90, 25)  # blue half
    mask = np.ones((SIZE, SIZE), dtype=np.uint8)
    info = extract_color(img, mask)
    assert info.name == "unknown"
    assert info.purity == 0.0


def test_tiny_mask_is_unknown():
    img, _ = solid_patch((36, 28, 221))
    mask = np.zeros((SIZE, SIZE), dtype=np.uint8)
    mask[:4, :4] = 1  # 16 px < min_pixels
    assert extract_color(img, mask).name == "unknown"


def test_rgb_roundtrip_close_to_input():
    info = extract_color(*solid_patch((25, 90, 200)))  # BGR → RGB (200, 90, 25)
    assert np.allclose(info.rgb, [200, 90, 25], atol=6)


@pytest.mark.parametrize(
    ("h", "s", "v", "expected"),
    [
        (358.0, 0.9, 0.8, "red"),  # red wraps past 360°
        (5.0, 0.9, 0.8, "red"),
        (30.0, 0.9, 0.9, "orange"),  # vivid+bright → orange, not wood
        (30.0, 0.4, 0.7, "wood"),  # same hue, washed out → wood
        (130.0, 0.7, 0.6, "green"),
        (0.0, 0.05, 0.95, "white"),
        (0.0, 0.05, 0.4, "grey"),
        (200.0, 0.9, 0.1, "black"),  # too dark to call blue
    ],
)
def test_name_from_hsv_rules(h, s, v, expected):
    assert name_from_hsv(h, s, v) == expected

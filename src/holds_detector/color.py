"""Dominant-color extraction from a hold's mask pixels (classical CV, no ML).

Pipeline per mask: erode (avoid wall-edge bleed) → drop shadow pixels →
k-means (k=3) in a hue-circular feature space → dominant cluster → name via
the centralized table below. Everything tunable lives in TUNING / HUE_TABLE.
"""

from __future__ import annotations

import math

import cv2
import numpy as np

from holds_detector.schema import ColorInfo

# ---------------------------------------------------------------------------
# Tunable tables — the single place to adjust color behavior (PRD F5).
# ---------------------------------------------------------------------------

# Hue (degrees, 0-360) → name. Checked in order; red wraps around 0.
HUE_TABLE = [
    ("red", 345.0, 360.0),
    ("red", 0.0, 10.0),
    ("orange", 10.0, 40.0),
    ("yellow", 40.0, 70.0),
    ("green", 70.0, 170.0),
    ("blue", 170.0, 260.0),
    ("purple", 260.0, 300.0),
    ("pink", 300.0, 345.0),
]

TUNING = {
    "erode_px": 3,  # mask shrink before sampling, avoids edge/wall bleed
    "shadow_v_max": 0.12,  # pixels darker than this are dropped as shadow
    "min_pixels": 30,  # fewer usable pixels than this → unknown
    "max_pixels": 20_000,  # subsample above this (k-means speed)
    "kmeans_k": 3,
    "purity_radius": 0.25,  # feature-space distance counted as "same color"
    "purity_min": 0.6,  # below this → unknown (multi-colored / uncertain)
    "black_v_max": 0.18,
    "white_v_min": 0.75,
    "grey_s_max": 0.20,  # below this saturation → black/white/grey by value
    "wood_hue": (20.0, 45.0),  # tan/brown band …
    "wood_s_max": 0.55,  # … counts as wood when not vividly saturated
    "wood_v_max": 0.55,  # … or when dark (brown = dark orange)
}

# Display colors (BGR) for the debug image outlines.
DISPLAY_BGR = {
    "red": (40, 30, 220),
    "orange": (20, 140, 250),
    "yellow": (30, 210, 240),
    "green": (60, 180, 60),
    "blue": (230, 130, 40),
    "purple": (200, 60, 150),
    "pink": (190, 110, 250),
    "black": (30, 30, 30),
    "white": (255, 255, 255),
    "grey": (140, 140, 140),
    "wood": (90, 140, 190),
    "unknown": (255, 0, 255),  # magenta = "look at me, I'm unsure"
}


def name_from_hsv(h: float, s: float, v: float) -> str:
    """Map one HSV value (h in degrees, s/v in 0-1) to a color name."""
    t = TUNING
    if v < t["black_v_max"]:
        return "black"
    if s < t["grey_s_max"]:
        return "white" if v >= t["white_v_min"] else "grey"
    lo, hi = t["wood_hue"]
    if lo <= h <= hi and (s < t["wood_s_max"] or v < t["wood_v_max"]):
        return "wood"
    for name, h_lo, h_hi in HUE_TABLE:
        if h_lo <= h < h_hi:
            return name
    return "unknown"


def _features(hsv: np.ndarray) -> np.ndarray:
    """HSV pixels → (s·cos h, s·sin h, v). Hue is circular (red sits at both
    0° and 360°), so we embed it on a circle scaled by saturation; plain
    k-means on raw hue would split red into two distant clusters."""
    h_rad = np.deg2rad(hsv[:, 0])
    s, v = hsv[:, 1], hsv[:, 2]
    return np.stack([s * np.cos(h_rad), s * np.sin(h_rad), v], axis=1).astype(np.float32)


def _unknown(rgb: list[int] | None = None, hsv: list[float] | None = None) -> ColorInfo:
    return ColorInfo(name="unknown", rgb=rgb or [0, 0, 0], hsv=hsv or [0.0, 0.0, 0.0], purity=0.0)


def extract_color(image_bgr: np.ndarray, mask: np.ndarray) -> ColorInfo:
    """Dominant color of the pixels of `image_bgr` selected by binary `mask`."""
    t = TUNING

    mask_u8 = (mask > 0).astype(np.uint8)
    eroded = cv2.erode(mask_u8, np.ones((3, 3), np.uint8), iterations=t["erode_px"])
    if cv2.countNonZero(eroded) < t["min_pixels"]:
        eroded = mask_u8  # tiny hold: erosion ate it, sample the full mask

    pixels_bgr = image_bgr[eroded.astype(bool)]
    if len(pixels_bgr) < t["min_pixels"]:
        return _unknown()

    # OpenCV full-range HSV: H 0-179 (half-degrees), S/V 0-255 → degrees & 0-1
    hsv_cv = cv2.cvtColor(pixels_bgr.reshape(1, -1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    hsv = hsv_cv.astype(np.float32) * np.array([2.0, 1 / 255.0, 1 / 255.0], dtype=np.float32)

    keep = hsv[:, 2] > t["shadow_v_max"]
    if keep.sum() >= t["min_pixels"]:  # don't drop everything on a black hold
        pixels_bgr, hsv = pixels_bgr[keep], hsv[keep]

    if len(hsv) > t["max_pixels"]:
        idx = np.random.default_rng(0).choice(len(hsv), t["max_pixels"], replace=False)
        pixels_bgr, hsv = pixels_bgr[idx], hsv[idx]

    feats = _features(hsv)
    k = min(t["kmeans_k"], len(feats))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 1e-3)
    _, labels, centers = cv2.kmeans(
        feats, k, None, criteria, attempts=3, flags=cv2.KMEANS_PP_CENTERS
    )
    labels = labels.ravel()

    dominant = np.bincount(labels, minlength=k).argmax()
    center = centers[dominant]
    # Purity counts ALL pixels near the dominant center, not just its cluster
    # members — k-means splits even a uniform color into k touching clusters,
    # which would make purity meaninglessly low for solid holds.
    dist = np.linalg.norm(feats - center, axis=1)
    purity = float((dist < t["purity_radius"]).mean())

    member_bgr = pixels_bgr[labels == dominant]
    mean_bgr = member_bgr.mean(axis=0)
    rgb = [int(round(c)) for c in mean_bgr[::-1]]

    # Cluster-center HSV: hue from the circular embedding (atan2), not a
    # naive mean, so red pixels at 359° and 1° average to 0° instead of 180°.
    s = float(np.hypot(center[0], center[1]))
    h = float(math.degrees(math.atan2(center[1], center[0]))) % 360.0
    v = float(center[2])
    hsv_out = [round(h, 1), round(s, 3), round(min(v, 1.0), 3)]

    if purity < t["purity_min"]:
        return _unknown(rgb=rgb, hsv=hsv_out)
    return ColorInfo(name=name_from_hsv(h, s, v), rgb=rgb, hsv=hsv_out, purity=round(purity, 3))

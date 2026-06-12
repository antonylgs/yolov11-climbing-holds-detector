"""Output data contract (PRD §6). Dataclasses + JSON serialization."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass
class ColorInfo:
    """Dominant color of one hold's mask pixels."""

    name: str  # one of COLOR_NAMES below
    rgb: list[int]  # [r, g, b] 0-255
    hsv: list[float]  # [h 0-360, s 0-1, v 0-1]
    purity: float  # fraction of mask pixels in the dominant color cluster


@dataclass
class Hold:
    id: int
    polygon: list[list[int]]  # [[x, y], ...] original-image pixels
    bbox: list[int]  # [x1, y1, x2, y2]
    center: list[int]  # [cx, cy] mask centroid
    area_px: int
    confidence: float
    color: ColorInfo
    hold_type: str | None = None  # reserved, always None in v1


@dataclass
class DetectionOutput:
    image: str
    image_size: list[int]  # [width, height]
    model: str
    holds: list[Hold] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent)


COLOR_NAMES = [
    "red",
    "orange",
    "yellow",
    "green",
    "blue",
    "purple",
    "pink",
    "black",
    "white",
    "grey",
    "wood",
    "unknown",
]

"""Generate the synthetic climbing-wall test fixture.

Draws a textured grey wall with colored hold-like blobs and saves it to
tests/fixtures/sample_wall.jpg. Deterministic (seeded), so the fixture can be
regenerated identically at any time. Run with:

    uv run python scripts/generate_fixture.py
"""

from pathlib import Path

import cv2
import numpy as np

WIDTH, HEIGHT = 960, 720
SEED = 42

# BGR colors roughly matching common hold colors
HOLD_COLORS = [
    (52, 34, 201),  # red
    (30, 140, 240),  # orange
    (40, 200, 230),  # yellow
    (70, 160, 40),  # green
    (190, 90, 30),  # blue
    (160, 50, 130),  # purple
    (170, 100, 240),  # pink
    (40, 40, 40),  # black
]


def make_wall(rng: np.random.Generator) -> np.ndarray:
    """Grey plywood-ish wall with noise texture and bolt holes."""
    wall = np.full((HEIGHT, WIDTH, 3), 165, dtype=np.uint8)
    noise = rng.normal(0, 8, size=(HEIGHT, WIDTH, 1)).astype(np.int16)
    wall = np.clip(wall.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # T-nut bolt-hole grid
    for y in range(60, HEIGHT, 120):
        for x in range(60, WIDTH, 120):
            cv2.circle(wall, (x, y), 3, (90, 90, 90), -1)
    return wall


def draw_hold(img: np.ndarray, rng: np.random.Generator, center: tuple[int, int]) -> None:
    """One hold: an irregular filled polygon with a darker outline and a highlight."""
    color = HOLD_COLORS[rng.integers(len(HOLD_COLORS))]
    radius = int(rng.integers(18, 45))
    angles = np.sort(rng.uniform(0, 2 * np.pi, size=int(rng.integers(7, 12))))
    radii = rng.uniform(0.6, 1.0, size=angles.shape) * radius
    pts = np.stack(
        [center[0] + radii * np.cos(angles), center[1] + radii * np.sin(angles)], axis=1
    ).astype(np.int32)
    shadow = tuple(int(c * 0.6) for c in color)
    cv2.fillPoly(img, [pts], color)
    cv2.polylines(img, [pts], isClosed=True, color=shadow, thickness=2)
    # small specular highlight so holds aren't flat color blobs
    hx, hy = center[0] - radius // 4, center[1] - radius // 4
    highlight = tuple(min(255, int(c * 1.4) + 30) for c in color)
    cv2.circle(img, (hx, hy), max(2, radius // 5), highlight, -1)


def main() -> None:
    rng = np.random.default_rng(SEED)
    wall = make_wall(rng)
    centers = [
        (int(rng.integers(60, WIDTH - 60)), int(rng.integers(60, HEIGHT - 60))) for _ in range(16)
    ]
    for center in centers:
        draw_hold(wall, rng, center)

    out = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "sample_wall.jpg"
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), wall, [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()

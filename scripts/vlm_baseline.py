"""Zero-shot vision-LLM baseline: ask Claude (or GPT) to box every climbing hold,
then score it with the SAME class-agnostic IoU scorer used in compare_baselines.py.

This is the "vs ChatGPT / Claude" comparison. General VLMs are not built for dense
small-object localization, so expect weak numbers — that is the intended evidence
that a task-specific trained model is required. Run on a SUBSET (cost!), e.g. --limit 60.

Setup:
  uv pip install anthropic        # or: openai
  export ANTHROPIC_API_KEY=...    # or OPENAI_API_KEY for --provider openai

Usage:
  uv run python scripts/vlm_baseline.py --limit 60 --provider anthropic \
      --model claude-sonnet-4-6 --out docs/eval

Predictions are cached to <out>/vlm_<model>_preds.json so re-scoring is free.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path

import cv2
import numpy as np

from _eval import evaluate, list_images, load_gt_boxes

PROMPT = (
    "This is a photo of an indoor climbing wall. Detect EVERY climbing hold "
    "(the bolted-on grips, of any color/shape). Return ONLY a JSON array, no prose. "
    "Each element: {\"box\": [x1, y1, x2, y2], \"confidence\": c} where x1,y1,x2,y2 are "
    "the bounding-box corners as fractions of image width/height in [0,1] (x1<x2, y1<y2) "
    "and c is your confidence in [0,1]. Detect as many holds as you can see."
)


def encode_image(path: Path, max_side: int) -> tuple[str, str, int, int]:
    """Downscale longest side to max_side; return (b64, media_type, orig_w, orig_h)."""
    img = cv2.imread(str(path))
    h, w = img.shape[:2]
    scale = min(1.0, max_side / max(h, w))
    if scale < 1.0:
        img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise RuntimeError(f"encode failed: {path}")
    return base64.b64encode(buf).decode(), "image/jpeg", w, h


def parse_boxes(text: str, w: int, h: int) -> np.ndarray:
    """Extract the JSON array from the reply → (N,5) [x1,y1,x2,y2,conf] in pixels."""
    s, e = text.find("["), text.rfind("]")
    if s == -1 or e == -1:
        return np.zeros((0, 5), dtype=np.float32)
    try:
        items = json.loads(text[s : e + 1])
    except json.JSONDecodeError:
        return np.zeros((0, 5), dtype=np.float32)
    rows: list[list[float]] = []
    for it in items:
        b = it.get("box") if isinstance(it, dict) else None
        if not b or len(b) != 4:
            continue
        x1, y1, x2, y2 = (float(v) for v in b)
        c = float(it.get("confidence", 1.0)) if isinstance(it, dict) else 1.0
        rows.append([min(x1, x2) * w, min(y1, y2) * h, max(x1, x2) * w, max(y1, y2) * h, c])
    return np.asarray(rows, dtype=np.float32).reshape(-1, 5)


def call_anthropic(b64: str, media: str, model: str) -> str:
    import anthropic

    client = anthropic.Anthropic()
    msg = client.messages.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": b64}},
                    {"type": "text", "text": PROMPT},
                ],
            }
        ],
    )
    return "".join(block.text for block in msg.content if block.type == "text")


def call_openai(b64: str, media: str, model: str) -> str:
    from openai import OpenAI

    client = OpenAI()
    resp = client.chat.completions.create(
        model=model,
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                ],
            }
        ],
    )
    return resp.choices[0].message.content or ""


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--test-dir", type=Path, default=Path("data/dataset/test"))
    p.add_argument("--provider", choices=["anthropic", "openai"], default="anthropic")
    p.add_argument("--model", default="claude-sonnet-4-6")
    p.add_argument("--out", type=Path, default=Path("docs/eval"))
    p.add_argument("--limit", type=int, default=60, help="subset size (VLM calls cost money)")
    p.add_argument("--max-side", type=int, default=1024, help="downscale longest image side")
    p.add_argument("--iou", type=float, default=0.5)
    p.add_argument("--pr-conf", type=float, default=0.30)
    p.add_argument("--refresh", action="store_true", help="ignore cache, re-query the API")
    args = p.parse_args(argv)

    key = "ANTHROPIC_API_KEY" if args.provider == "anthropic" else "OPENAI_API_KEY"
    if not os.environ.get(key):
        raise SystemExit(f"set {key} in the environment")

    images = list_images(args.test_dir / "images", args.limit)
    if not images:
        raise SystemExit(f"no images in {args.test_dir / 'images'}")
    args.out.mkdir(parents=True, exist_ok=True)
    cache_path = args.out / f"vlm_{args.model.replace('/', '_')}_preds.json"
    cache: dict[str, list] = {}
    if cache_path.exists() and not args.refresh:
        cache = json.loads(cache_path.read_text())
    call = call_anthropic if args.provider == "anthropic" else call_openai

    preds: dict[str, np.ndarray] = {}
    gt: dict[str, np.ndarray] = {}
    for i, img in enumerate(images, 1):
        h, w = cv2.imread(str(img)).shape[:2]
        gt[img.stem] = load_gt_boxes(args.test_dir / "labels" / f"{img.stem}.txt", w, h)
        if img.stem in cache:
            preds[img.stem] = np.asarray(cache[img.stem], dtype=np.float32).reshape(-1, 5)
            continue
        b64, media, _, _ = encode_image(img, args.max_side)
        try:
            arr = parse_boxes(call(b64, media, args.model), w, h)
        except Exception as exc:  # noqa: BLE001 - one bad image shouldn't kill the run
            print(f"  ! {img.name}: {exc}")
            arr = np.zeros((0, 5), dtype=np.float32)
        preds[img.stem] = arr
        cache[img.stem] = arr.tolist()
        cache_path.write_text(json.dumps(cache))  # checkpoint after every call
        print(f"  [{i}/{len(images)}] {img.name}: {len(arr)} boxes", flush=True)

    m = evaluate(preds, gt, iou_thr=args.iou, pr_conf=args.pr_conf)
    md = (
        f"# VLM zero-shot baseline — {args.provider}:{args.model}\n\n"
        f"Subset: {len(images)} images, {int(m['n_gt'])} holds. "
        f"Class-agnostic, IoU={args.iou}, P/R @ conf≥{args.pr_conf}.\n\n"
        "| Model | mAP@50 | Precision | Recall | F1 |\n|---|---|---|---|---|\n"
        f"| {args.provider}:{args.model} | {m['ap50']:.3f} | "
        f"{m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} |\n\n"
        "> Note: VLM confidence is poorly calibrated, so treat P/R/F1 as the primary "
        "signal; mAP is indicative only.\n"
    )
    (args.out / f"vlm_{args.model.replace('/', '_')}.md").write_text(md)
    print("\n" + md)
    print(f"cache: {cache_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

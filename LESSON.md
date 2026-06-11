# Lesson: Fine-tuning a Model to Detect Climbing Holds

Dead-simple explanation of what we're doing and why. Read top to bottom.

---

## 1. The goal

Give the computer a photo of a climbing wall → get back, for every hold:

- **Where it is** (exact outline, as a polygon of x,y coordinates)
- **What color it is** (e.g. "red", with the raw RGB value)
- **Extras**: size in pixels, center point, and a confidence score (how sure the model is)

## 2. What "fine-tuning" actually means

Training a vision model from scratch needs millions of images and weeks of GPU time. Nobody does that for a project like this.

Instead, we take a model that **already knows how to see** — it was pre-trained on millions of everyday photos and understands edges, textures, shapes, objects — and we **continue its training** on a small dataset of climbing wall photos. The model keeps its general vision skills and *specializes* in our task. That's fine-tuning.

Analogy: hiring an experienced photographer and teaching them what a climbing hold looks like, vs. raising a baby and teaching it to see from zero.

## 3. Detection vs. segmentation (why we chose segmentation)

- **Object detection** = draw a rectangle (bounding box) around each hold.
- **Instance segmentation** = trace the exact outline (mask) of each hold.

We chose segmentation because holds are irregular blobs. A rectangle around a hold contains lots of wall pixels — if we then ask "what color is inside this rectangle?" we'd get polluted answers. With an exact outline, we only look at hold pixels, so color extraction is clean.

## 4. The model: YOLO

**YOLO** ("You Only Look Once") is the standard family of models for fast object detection/segmentation. We'll use **Ultralytics YOLO11-seg** (or YOLOv8-seg) because:

- It has a segmentation variant out of the box
- Fine-tuning it is ~5 lines of Python
- Small versions run fine on a laptop
- Huge community, tons of tutorials

Model sizes go nano → small → medium → large. We start with **small (`yolo11s-seg`)**: good accuracy, fast enough on your Mac.

## 5. The secret trick: don't make the model learn color

Naive approach: create classes like `red_hold`, `blue_hold`, `green_hold`... and label thousands of examples of each. Problems: way more labeling work, and the model gets confused by gym lighting, shadows, faded holds.

Smarter approach (ours): the model learns **one class only — "hold"** — and just finds outlines. Then a tiny piece of classical (non-AI) code looks at the actual pixels inside each outline and computes the dominant color. This is:

- More accurate (we read real pixel values)
- Zero extra labeling
- Easy to tweak (changing color names doesn't require retraining)

**Rule of thumb you just learned:** only make the neural network do what classical code can't.

## 6. The data: what the model learns from

Fine-tuning needs labeled examples: photos + their "answer key" (the outlines a human drew). Our plan:

1. **Public datasets**: Roboflow Universe hosts free community datasets of labeled climbing holds (thousands of images). We start with these.
2. **Your own photos**: ~50–150 photos of your gym, which you label yourself. This adapts the model to *your* walls, lighting, and hold styles.

We split data into three buckets:

- **Train (~80%)** — the model learns from these
- **Validation (~10%)** — checked during training to detect overfitting
- **Test (~10%)** — touched only once at the end, for an honest final score

**Overfitting** = the model memorizes the training photos instead of learning the general concept "hold". Symptom: great scores on training images, bad scores on new ones. The validation set is our smoke alarm for this.

## 7. Labeling

Labeling = drawing polygons around every hold in a photo, by hand, in a tool (Roboflow's web annotator, or CVAT, or Label Studio). It's tedious but it IS the project — model quality is capped by label quality.

Shortcut we'll use: **model-assisted labeling**. Train a first model on public data → let it pre-draw outlines on your photos → you just fix its mistakes. 5–10× faster than from scratch.

## 8. Training: what actually happens

On Google Colab (free cloud GPU), we run roughly:

```python
from ultralytics import YOLO
model = YOLO("yolo11s-seg.pt")          # pre-trained weights
model.train(data="dataset.yaml", epochs=100, imgsz=1280)
```

What happens inside, per **epoch** (= one full pass over all training images):

1. Model predicts outlines on a batch of images
2. A **loss function** measures how wrong it was vs. your labels
3. The model's internal numbers (weights) get nudged to be slightly less wrong
4. Repeat for all batches, then check score on validation set

After ~50–150 epochs the validation score plateaus → done. Output: a `best.pt` file (~20 MB) — that's your fine-tuned model.

Terms you'll see:
- **batch size**: images processed at once (limited by GPU memory)
- **imgsz**: resolution images are resized to. Higher = sees small holds better, slower. Wall photos have many small holds → we use 1280, larger than the default 640.
- **learning rate**: how big each "nudge" is. Defaults are fine; don't touch initially.
- **augmentation**: random flips/brightness/crops applied to training images so the model sees more variety for free. Ultralytics does this automatically.

## 9. Measuring success

- **mAP50** (main metric): roughly "what % of holds does it find correctly, allowing loose outline overlap". 0.85+ is good for this task.
- **mAP50-95**: same but demanding tighter outlines. Always lower; ~0.6+ is solid.
- **Precision**: of everything it called a hold, how many really were (low = false alarms, e.g. detecting chalk marks).
- **Recall**: of all real holds, how many it found (low = missed holds).

But honestly: the best eval at the start is **looking at predictions on test photos with your eyes**.

## 10. Color extraction (the classical CV part)

For each detected mask:

1. Take only the pixels inside the outline
2. Convert to **HSV** color space (Hue/Saturation/Value) — better than RGB for naming colors because "what color" lives mostly in the single Hue number, regardless of lighting brightness
3. Find the dominant color (cluster the pixels, take the biggest cluster — ignores shadows/chalk spots)
4. Map hue ranges to names: red, orange, yellow, green, blue, purple, pink, black, white, grey

## 11. The full pipeline, end to end

```
photo.jpg
   │
   ▼
YOLO11s-seg (fine-tuned)  ──►  N masks + confidence scores
   │
   ▼
per mask: pixel analysis  ──►  dominant color, RGB, name
   │
   ▼
JSON output:
[
  { "id": 1, "polygon": [[x,y],...], "bbox": [x1,y1,x2,y2],
    "center": [x,y], "area_px": 1234,
    "color": {"name": "red", "rgb": [201,34,52]},
    "confidence": 0.91 },
  ...
]
+ a debug image with outlines drawn on it
```

## 12. The whole project as steps

1. Set up environment (Python, Ultralytics) — 30 min
2. Get public dataset from Roboflow, train **baseline model** on Colab — 1 evening
3. Eval baseline on YOUR gym photos → see where it fails
4. Take 50–150 photos of your gym, label them (model-assisted) — the grind, a few evenings
5. Re-train on public + your data → better model
6. Build the color-extraction + JSON output script
7. Iterate: find failure cases → add/fix data → retrain (this loop is 80% of real ML work)

## 13. Mistakes beginners make (so you don't)

- **Too little/too messy data, too much hyperparameter fiddling.** Data quality beats settings tuning 10:1.
- **Inconsistent labels** (sometimes labeling volumes, sometimes not; cutting off hold edges). Decide rules upfront, write them down.
- **Judging only by metrics.** Always look at actual predictions.
- **Testing on training images.** Always keep the test set untouched.
- **Training color into the model** — covered above, we're not doing it. 😉

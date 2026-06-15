"""Fine-tune a YOLO segmentation model on the prepared holds dataset.

One training code path, two execution environments: the owner's Mac (MPS/CPU) and
Google Colab (CUDA). The Colab notebook shells out to ``scripts/train.py``, which
calls :func:`train` here — there is no second copy of the training logic.

Project defaults are fixed by PRD §4: base weights ``yolo11s-seg.pt``, ``imgsz``
1280, single class ``hold``. Everything is overridable for the local smoke run
(``--epochs 2 --imgsz 320``) and for tuning later (issue 009).

Ultralytics already records the full run for reproducibility — ``args.yaml`` in the
run directory captures the data config, base weights, and every hyperparameter.
We verify that survives and add a small ``run_meta.json`` with the resolved device
and the dataset's split seed, so a run folder is self-describing without guessing.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

DEFAULT_BASE_WEIGHTS = "yolo11s-seg.pt"  # PRD §4; auto-downloaded by ultralytics
DEFAULT_EPOCHS = 100
DEFAULT_IMGSZ = 1280
DEFAULT_BATCH = 16
DEFAULT_PROJECT = "runs"  # gitignored; ultralytics auto-increments name → runs/holds, runs/holds2…
DEFAULT_NAME = "holds"

_ROOT = Path(__file__).resolve().parents[2]
_MODELS_DIR = _ROOT / "models"


def select_device(explicit: str | None = None) -> str:
    """Pick a torch device. An explicit choice always wins; otherwise prefer CUDA
    (Colab), then Apple MPS (the owner's laptop), then CPU. Returns an
    ultralytics-friendly string: ``"0"`` for the first GPU, ``"mps"``, or ``"cpu"``."""
    if explicit:
        return explicit
    import torch  # deferred: heavy import, only needed at train time

    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_resume(resume: str | bool, project: str) -> Path:
    """Locate the checkpoint to resume from. A path resumes that run; ``True``
    finds the most recently modified ``last.pt`` under ``project/`` — the common
    case after a Colab session is killed mid-run."""
    if isinstance(resume, str):
        ckpt = Path(resume)
        if not ckpt.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt}")
        return ckpt
    candidates = sorted(
        (_ROOT / project).glob("**/weights/last.pt"),
        key=lambda p: p.stat().st_mtime,
    )
    if not candidates:
        raise FileNotFoundError(
            f"no last.pt to resume under {project}/ — start a fresh run instead"
        )
    return candidates[-1]


def train(
    data: str | Path,
    *,
    base_weights: str | Path = DEFAULT_BASE_WEIGHTS,
    epochs: int = DEFAULT_EPOCHS,
    imgsz: int = DEFAULT_IMGSZ,
    batch: int = DEFAULT_BATCH,
    device: str | None = None,
    project: str = DEFAULT_PROJECT,
    name: str = DEFAULT_NAME,
    resume: str | bool = False,
    export_best: bool = True,
    extra: dict | None = None,
) -> Path:
    """Fine-tune segmentation weights and return the path to the run's ``best.pt``.

    When ``resume`` is set, ultralytics reloads the checkpoint and continues with
    that run's stored args — ``data``/``epochs``/etc. are ignored by design, so a
    resumed run can never silently diverge from the one it's continuing.

    With ``export_best`` (default) the winning weights are also copied to
    ``models/best.pt`` so the inference CLI (issue 002) picks them up automatically.
    """
    from ultralytics import YOLO  # deferred: heavy import, tests stub the train path

    device = select_device(device)

    if resume:
        ckpt = _resolve_resume(resume, project)
        print(f"resuming from {ckpt} on device {device}")
        model = YOLO(str(ckpt))
        model.train(resume=True)
    else:
        print(f"training {base_weights} on {data} | imgsz={imgsz} epochs={epochs} device={device}")
        model = YOLO(str(base_weights))
        model.train(
            data=str(data),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=device,
            project=project,
            name=name,
            single_cls=True,  # PRD §4: one class, regardless of source label ids
            verbose=False,
            **(extra or {}),
        )

    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():  # ultralytics writes best.pt only when a val run improves
        best = save_dir / "weights" / "last.pt"

    _write_run_meta(
        save_dir, data=data, base_weights=base_weights, device=device, resumed=bool(resume)
    )

    if export_best:
        _MODELS_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(best, _MODELS_DIR / "best.pt")
        print(f"exported {best.name} → {_MODELS_DIR / 'best.pt'}")

    print(f"done: {best}")
    return best


def _write_run_meta(
    save_dir: Path, *, data: str | Path, base_weights: str | Path, device: str, resumed: bool
) -> None:
    """Drop a small, human-readable provenance file beside ultralytics' args.yaml.

    Ultralytics already records the hyperparameters; what it doesn't surface in one
    place is the dataset's split seed (lives in the dataset's validation_report.json,
    from issue 003). Pulling it in makes 'which exact data trained this?' a one-file
    answer — the reproducibility hook for comparing model v1 vs v2."""
    meta: dict = {
        "base_weights": str(base_weights),
        "data_config": str(data),
        "device": device,
        "resumed": resumed,
    }
    vr = _validation_report_for(Path(data))
    if vr and vr.exists():
        meta["dataset_report"] = str(vr)
        try:
            meta["dataset_seed"] = json.loads(vr.read_text()).get("seed")
        except (ValueError, OSError):
            pass

    (save_dir / "run_meta.json").write_text(json.dumps(meta, indent=2) + "\n")


def _validation_report_for(data_config: Path) -> Path | None:
    """The dataset's validation_report.json lives in the dataset root, which the
    config names via its ``path:`` field (issue 003 emits it there)."""
    try:
        for line in data_config.read_text().splitlines():
            if line.startswith("path:"):
                root = Path(line.split(":", 1)[1].strip())
                if not root.is_absolute():
                    root = (data_config.resolve().parent / root).resolve()
                return root / "validation_report.json"
    except OSError:
        pass
    return None

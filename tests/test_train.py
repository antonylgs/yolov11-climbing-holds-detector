"""Tests for the training entrypoint (scripts/train.py / holds_detector.train).

Device selection is checked without a GPU; the real training path is exercised
once on the fixture dataset (nano weights, 2 epochs, 96 px, CPU) — a machinery
check that the same code the Colab notebook calls produces a best.pt and a
self-describing run folder. No downloads beyond the cached nano checkpoint."""

import json
from pathlib import Path

import pytest

from holds_detector import train as train_mod
from holds_detector.dataset import prepare

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "dataset"
GOOD_EXPORTS = [FIXTURES / "export_a", FIXTURES / "export_b"]
MODEL_PATH = ROOT / "models" / "yolo11n-seg.pt"


def test_defaults_match_prd() -> None:
    # PRD §4: yolo11s-seg base, imgsz 1280; long-haul epoch budget.
    assert train_mod.DEFAULT_BASE_WEIGHTS == "yolo11s-seg.pt"
    assert train_mod.DEFAULT_IMGSZ == 1280
    assert train_mod.DEFAULT_EPOCHS == 100


def test_select_device_honors_explicit() -> None:
    assert train_mod.select_device("cpu") == "cpu"
    assert train_mod.select_device("mps") == "mps"


def test_select_device_auto_returns_valid() -> None:
    assert train_mod.select_device(None) in {"0", "mps", "cpu"}


def test_resume_with_no_runs_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        train_mod._resolve_resume(True, project=str(tmp_path / "empty"))


def test_resume_missing_path_raises() -> None:
    with pytest.raises(FileNotFoundError):
        train_mod._resolve_resume("/nope/last.pt", project="runs")


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_train_smoke_produces_best_and_provenance(tmp_path: Path, monkeypatch) -> None:
    """Acceptance criterion (local): --epochs 2 --imgsz 320-class run completes on
    CPU and produces a best.pt. Run at 96 px to keep the test fast; the path is
    identical."""
    # keep models/best.pt export inside the tmp dir so the test never clobbers a real model
    monkeypatch.setattr(train_mod, "_MODELS_DIR", tmp_path / "models")

    prepare(
        sources=GOOD_EXPORTS,
        out_dir=tmp_path / "dataset",
        config_path=tmp_path / "dataset.yaml",
    )

    best = train_mod.train(
        data=tmp_path / "dataset.yaml",
        base_weights=MODEL_PATH,
        epochs=2,
        imgsz=96,
        batch=4,
        device="cpu",
        project=str(tmp_path / "runs"),
        name="smoke",
        extra={"workers": 0, "plots": False},
    )

    assert best.exists() and best.suffix == ".pt"

    save_dir = best.parent.parent
    # Ultralytics records the run for reproducibility; verify it survives.
    args_yaml = (save_dir / "args.yaml").read_text()
    assert "data:" in args_yaml
    assert "epochs: 2" in args_yaml

    # Our augmentation: device + the dataset's split seed in one place.
    meta = json.loads((save_dir / "run_meta.json").read_text())
    assert meta["device"] == "cpu"
    assert meta["dataset_seed"] == 42

    # export_best copied the winner where the inference CLI looks for it.
    assert (tmp_path / "models" / "best.pt").exists()

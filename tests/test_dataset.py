"""Tests for dataset preparation (scripts/prepare_dataset.py / holds_detector.dataset).

Run entirely against the checked-in fixture exports in tests/fixtures/dataset/ —
no downloads. The final test fine-tunes for one epoch on the fixture to prove the
emitted dataset.yaml is genuinely consumable by Ultralytics."""

import hashlib
import json
from pathlib import Path

import pytest

from holds_detector.dataset import main, prepare, shoelace_area

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "dataset"
GOOD_EXPORTS = [FIXTURES / "export_a", FIXTURES / "export_b"]
DEFECT_EXPORT = FIXTURES / "export_defects"
MODEL_PATH = ROOT / "models" / "yolo11n-seg.pt"

N_GOOD_IMAGES = 10  # export_a: 4 train + 2 valid; export_b: 4 train


def run_prepare(tmp_path: Path, sources, **kwargs) -> dict:
    return prepare(
        sources=list(sources),
        out_dir=tmp_path / "dataset",
        config_path=tmp_path / "dataset.yaml",
        **kwargs,
    )


def tree_hashes(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def all_label_lines(dataset_dir: Path) -> list[str]:
    return [
        line
        for label_file in sorted(dataset_dir.rglob("labels/*.txt"))
        for line in label_file.read_text().splitlines()
    ]


# --- determinism -----------------------------------------------------------


def test_same_seed_is_byte_identical(tmp_path: Path) -> None:
    run_prepare(tmp_path, GOOD_EXPORTS, seed=42)
    first = tree_hashes(tmp_path)
    run_prepare(tmp_path, GOOD_EXPORTS, seed=42)  # out dir is wiped and rebuilt
    assert tree_hashes(tmp_path) == first


def test_different_seed_changes_split(tmp_path: Path) -> None:
    # any single pair of seeds may collide by chance (1 val slot, 10 images),
    # so assert variety across a handful of seeds instead
    memberships = set()
    sizes = set()
    for seed in range(5):
        report = run_prepare(tmp_path, GOOD_EXPORTS, seed=seed)
        sizes.add(tuple(sorted(report["splits"].items())))
        val = tuple(sorted(p.name for p in (tmp_path / "dataset" / "val" / "images").iterdir()))
        test = tuple(sorted(p.name for p in (tmp_path / "dataset" / "test" / "images").iterdir()))
        memberships.add((val, test))
    assert len(sizes) == 1  # sizes never change
    assert len(memberships) > 1  # membership reshuffles


# --- merging and remapping -------------------------------------------------


def test_merges_sources_into_single_class(tmp_path: Path) -> None:
    report = run_prepare(tmp_path, GOOD_EXPORTS)
    assert report["images_valid"] == N_GOOD_IMAGES
    lines = all_label_lines(tmp_path / "dataset")
    assert lines, "no labels written"
    # export_a contains class ids 0 AND 1; everything must come out as class 0
    assert all(line.split()[0] == "0" for line in lines)
    config = (tmp_path / "dataset.yaml").read_text()
    assert "0: hold" in config and "nc: 1" in config


def test_split_is_80_10_10_by_image(tmp_path: Path) -> None:
    report = run_prepare(tmp_path, GOOD_EXPORTS)
    assert report["splits"] == {"train": 8, "val": 1, "test": 1}
    for split in ("train", "val", "test"):
        images = list((tmp_path / "dataset" / split / "images").iterdir())
        labels = list((tmp_path / "dataset" / split / "labels").iterdir())
        assert len(images) == len(labels) == report["splits"][split]


def test_same_stem_in_different_splits_does_not_collide(tmp_path: Path) -> None:
    # export_a has wall_0 in both train/ and valid/ — both must survive the merge
    run_prepare(tmp_path, [FIXTURES / "export_a"])
    stems = [p.stem for p in (tmp_path / "dataset").rglob("images/*.jpg")]
    assert len(stems) == len(set(stems)) == 6


def test_safe_stem_caps_length_and_stays_unique() -> None:
    # Roboflow-style monster stems must not exceed the 255-byte filename limit, and
    # two that share a long common prefix must still produce distinct outputs.
    from holds_detector.dataset import _MAX_STEM_LEN, _safe_stem

    base = "climbing-hold-" + "x" * 300
    a = _safe_stem(base + "_aaa")
    b = _safe_stem(base + "_bbb")
    assert len(a) <= _MAX_STEM_LEN and len(b) <= _MAX_STEM_LEN
    assert a != b
    assert _safe_stem("short_name") == "short_name"  # short names pass through


# --- validation ------------------------------------------------------------


def test_validation_catches_each_defect_type(tmp_path: Path) -> None:
    report = run_prepare(tmp_path, [DEFECT_EXPORT])
    kinds = {
        finding["kind"] for entry in report["quarantined"] for finding in entry["findings"]
    }
    assert kinds == {"missing_label", "empty_label", "too_few_points", "zero_area", "out_of_bounds"}
    assert report["images_valid"] == 0
    quarantined_images = list((tmp_path / "dataset" / "quarantine").glob("*.jpg"))
    assert len(quarantined_images) == 5
    # report is also persisted next to the dataset
    on_disk = json.loads((tmp_path / "dataset" / "validation_report.json").read_text())
    assert on_disk["quarantined"] == report["quarantined"]


def test_strict_mode_fails_and_writes_nothing(tmp_path: Path) -> None:
    exit_code = main(
        [
            str(DEFECT_EXPORT),
            "--out",
            str(tmp_path / "dataset"),
            "--config",
            str(tmp_path / "dataset.yaml"),
            "--strict",
        ]
    )
    assert exit_code == 1
    assert not (tmp_path / "dataset").exists()
    assert not (tmp_path / "dataset.yaml").exists()


def test_strict_mode_passes_on_clean_data(tmp_path: Path) -> None:
    report = run_prepare(tmp_path, GOOD_EXPORTS, strict=True)
    assert report["ok"] and report["images_valid"] == N_GOOD_IMAGES


def test_shoelace_area() -> None:
    assert shoelace_area([(0, 0), (1, 0), (1, 1), (0, 1)]) == pytest.approx(1.0)
    assert shoelace_area([(0.1, 0.1), (0.5, 0.5), (0.9, 0.9)]) == 0.0  # collinear


# --- the emitted config actually trains ------------------------------------


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_emitted_config_trains_one_epoch(tmp_path: Path) -> None:
    """Acceptance criterion: Ultralytics can consume dataset.yaml end to end.
    One epoch, nano model, 96 px — machinery check, not a quality check."""
    from ultralytics import YOLO

    run_prepare(tmp_path, GOOD_EXPORTS)
    model = YOLO(str(MODEL_PATH))  # downloaded on first use, cached in models/
    model.train(
        data=str(tmp_path / "dataset.yaml"),
        epochs=1,
        imgsz=96,
        batch=4,
        device="cpu",
        workers=0,
        project=str(tmp_path / "runs"),
        name="smoke",
        verbose=False,
        plots=False,
    )
    assert (tmp_path / "runs" / "smoke" / "weights" / "last.pt").exists()

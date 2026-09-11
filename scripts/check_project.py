"""Check a checkout's Python, model files and catalogue without training anything.

Run with the same interpreter used for the API:
    .venv/Scripts/python.exe scripts/check_project.py --load-models
Git freshness is separate: git fetch origin, then git status -sb.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LFS_HEADER = b"version https://git-lfs.github.com/spec/v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-models", action="store_true",
                        help="also load all API models and validate the search index")
    args = parser.parse_args()
    failures = []

    def report(ok, label, detail="", optional=False):
        status = "OK" if ok else "NOTE" if optional else "FAIL"
        print(f"[{status}] {label}" + (f": {detail}" if detail else ""))
        if not ok and not optional:
            failures.append(label)

    def check_file(relative):
        path = ROOT / relative
        if not path.is_file():
            report(False, relative, "missing; restore the tracked file / dataset")
            return False
        with path.open("rb") as handle:
            prefix = handle.read(128)
        if not prefix or prefix.startswith(LFS_HEADER):
            report(False, relative, "empty or Git LFS pointer; run git lfs pull")
            return False
        report(True, relative)
        return True

    print(f"Project: {ROOT}\nPython: {sys.executable} ({sys.version.split()[0]})")
    dependencies = {
        "torch": "torch", "numpy": "numpy", "pandas": "pandas", "PIL": "Pillow",
        "scipy": "scipy", "sklearn": "scikit-learn", "joblib": "joblib",
        "fastapi": "fastapi", "uvicorn": "uvicorn", "multipart": "python-multipart",
    }
    for module, package in dependencies.items():
        report(importlib.util.find_spec(module) is not None, package,
               "install requirements-backend.txt with this interpreter"
               if importlib.util.find_spec(module) is None else "")
    for module in ("pytest", "pip", "cv2", "rembg"):
        report(importlib.util.find_spec(module) is not None, module,
               "optional for serving; see README setup", optional=True)

    for relative in (
        "artifacts/task1_120x160/task1_120x160_background_adapted.pt",
        "artifacts/task2/task2_season_best_pytorch.pth",
        "artifacts/task2/task2_season_class_mapping.json",
        "artifacts/task3/task3_cnn_model.pt",
        "artifacts/task2b_sfs/task2b_sfs_logreg.joblib",
        "artifacts/task2b_sfs/task1_to_sfs_mapping.json",
        "artifacts/task3b_recommendation/task3b_recommendation_priors.json",
        "artifacts/task4/gallery_metadata.csv",
        "artifacts/task4/notebook_report_outputs.json.gz",
        "app/frontend/index.html",
    ):
        check_file(relative)
    if check_file("artifacts/task4/search_manifest.json"):
        try:
            manifest = json.loads((ROOT / "artifacts/task4/search_manifest.json").read_text())
            for key in ("index_file", "encoder_file"):
                check_file("artifacts/task4/" + manifest[key])
        except (KeyError, ValueError, TypeError) as error:
            report(False, "Task 4 manifest", str(error))

    dataset = "A2_FashionDataset/FashionDataset"
    for split, metadata in (("train", "styles_train.csv"),
                            ("test", "styles_prediction_template.csv")):
        relative = f"{dataset}/{split}/{metadata}"
        if not check_file(relative):
            continue
        # Raw training metadata includes five records without usable images.
        # Check the catalogue actually served, which excludes them during EDA.
        coverage = (ROOT / "artifacts/task4/gallery_metadata.csv"
                    if split == "train" else ROOT / relative)
        if not coverage.is_file():
            continue  # already reported by check_file above
        with coverage.open(encoding="utf-8-sig", newline="") as handle:
            ids = {row["id"] for row in csv.DictReader(handle)}
        folder = ROOT / dataset / split / f"images_{split}"
        bad = []
        for item_id in ids:
            path = folder / f"{item_id}.jpg"
            if not path.is_file():
                bad.append(item_id)
            elif path.stat().st_size < 256:
                with path.open("rb") as handle:
                    prefix = handle.read(128)
                if not prefix or prefix.startswith(LFS_HEADER):
                    bad.append(item_id)
        report(not bad, f"{split} images", f"{len(ids) - len(bad)}/{len(ids)} present"
               + (f"; missing/empty/pointers: {bad[:5]}" if bad else ""))

    for relative, rebuild in (
        ("A2_FashionDataset/processed/clean_train_metadata.csv", "notebooks/01_eda.ipynb"),
        ("A2_FashionDataset/processed/image_cache_task1_60x80.npy", "scripts/build_task1_cache.py"),
        ("A2_FashionDataset/processed/image_cache_task1_60x80_ids.npy", "scripts/build_task1_cache.py"),
        ("artifacts/task1_120x160/task1_predictions_120x160.csv", "scripts/refresh_task1_fixture.py"),
    ):
        report((ROOT / relative).is_file(), relative, f"rebuild with {rebuild}", optional=True)

    if args.load_models and not failures:
        try:
            from app.backend import main as backend
            backend.load_models()
            for name in ("task1", "task2", "task2b", "task3", "task3b", "task4"):
                report(getattr(backend, f"{name}_service") is not None, f"load {name}",
                       getattr(backend, f"{name}_error") or "")
        except Exception as error:
            report(False, "API model loading", f"{type(error).__name__}: {error}")
    print(f"\n{len(failures)} required check(s) failed.")
    return int(bool(failures))


if __name__ == "__main__":
    raise SystemExit(main())

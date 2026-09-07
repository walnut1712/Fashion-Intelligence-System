"""The graded submission: shape, completeness, and which recipe produced it.

Nothing in ``tests/`` covered ``outputs/predictions/styles_prediction.csv`` or
``scripts/build_submission.py``, which left the deliverable itself the one
artefact with no regression test. That gap is not theoretical: the submission
column and ``outputs/task1_item_type_predictions_prior_corrected.csv`` disagree
on 1,024 of 5,829 rows, and with no provenance recorded anywhere, deciding which
of them was the shipped recipe took a full re-run of the inference path.

``build_submission.py`` already asserts the shape invariants internally, but only
while it is running. These tests assert them about the file that is actually on
disk, which is what gets marked.
"""

import json
import re
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SUBMISSION = PROJECT_ROOT / "outputs" / "predictions" / "styles_prediction.csv"
TASK1_COLUMN = PROJECT_ROOT / "outputs" / "task1_item_type_predictions.csv"
RECIPE_MANIFEST = TASK1_COLUMN.with_suffix(TASK1_COLUMN.suffix + ".recipe.json")
TEMPLATE = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / "test"
            / "styles_prediction_template.csv")
SUMMARY = PROJECT_ROOT / "artifacts" / "task1" / "task1_summary.json"

TARGETS = ("gender", "articleType", "season", "usage")


@pytest.fixture(scope="module")
def pandas():
    return pytest.importorskip("pandas")


@pytest.fixture(scope="module")
def submission(pandas):
    if not SUBMISSION.exists():
        pytest.skip("{} not built".format(SUBMISSION))
    return pandas.read_csv(SUBMISSION)


@pytest.fixture(scope="module")
def template(pandas):
    if not TEMPLATE.exists():
        pytest.skip("dataset not present (gitignored)")
    return pandas.read_csv(TEMPLATE)


def test_submission_matches_the_template_shape(submission, template):
    """Same columns, same order, same ids in the same order."""
    assert list(submission.columns) == list(template.columns)
    assert len(submission) == len(template)
    assert list(submission["id"]) == list(template["id"])


def test_submission_has_no_empty_cells(submission):
    """Every target filled - a blank cell is an unanswerable row, not a guess."""
    for column in TARGETS:
        missing = int(submission[column].isna().sum())
        assert missing == 0, "{} has {} empty cells".format(column, missing)


def test_submission_articletype_comes_from_the_task1_column(submission, pandas):
    """The join must not reorder or resample what Task 1 predicted."""
    if not TASK1_COLUMN.exists():
        pytest.skip("{} not built".format(TASK1_COLUMN))
    task1 = pandas.read_csv(TASK1_COLUMN)
    assert list(task1["id"]) == list(submission["id"])
    assert list(task1["articleType"]) == list(submission["articleType"])


def test_task1_column_matches_its_recipe_manifest(pandas):
    """The column on disk is the one the recorded recipe actually produced.

    ``predict.py --submission`` writes the manifest beside the CSV. If someone
    regenerates the column by another route - a single checkpoint, a different
    alpha, no correction at all - the digest stops matching and this fails,
    rather than the two silently diverging with nothing on disk to say so.
    """
    if not RECIPE_MANIFEST.exists():
        pytest.skip("no recipe manifest; regenerate with predict.py --submission")
    import hashlib

    manifest = json.loads(RECIPE_MANIFEST.read_text(encoding="utf-8"))
    labels = pandas.read_csv(TASK1_COLUMN)["articleType"]
    digest = hashlib.sha256("\n".join(map(str, labels)).encode("utf-8")).hexdigest()

    assert manifest["rows"] == len(labels)
    assert manifest["articletype_sha256"] == digest, (
        "the submission column has changed since the recipe manifest was written")


def test_recipe_manifest_matches_the_documented_pipeline():
    """What ran matches what ``task1_summary.json`` says ships.

    This is the check that would have settled the ambiguity immediately: the
    summary claims an ensemble plus a half-strength label-shift correction, and
    this asserts the file was in fact built that way.
    """
    if not RECIPE_MANIFEST.exists():
        pytest.skip("no recipe manifest; regenerate with predict.py --submission")
    if not SUMMARY.exists():
        pytest.skip("{} not present".format(SUMMARY))

    manifest = json.loads(RECIPE_MANIFEST.read_text(encoding="utf-8"))
    documented = json.loads(SUMMARY.read_text(encoding="utf-8")).get(
        "submission_pipeline") or {}

    expected_models = documented.get("submission_ensemble")
    if expected_models:
        assert manifest["models"] == list(expected_models)

    correction = documented.get("label_shift_correction") or ""
    match = re.search(r"alpha\s*=\s*([0-9.]+)", correction)
    if match:
        assert manifest["prior_correct"] is True
        assert manifest["alpha"] == pytest.approx(float(match.group(1)))

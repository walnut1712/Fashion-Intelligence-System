"""Recorded evidence survives reruns without executing archived output content."""

import copy
import gzip
import json
from pathlib import Path
import sys

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.notebook_report import RecordedNotebookReport


@pytest.fixture
def archive(tmp_path):
    def write(payload):
        path = tmp_path / "report.json.gz"
        with gzip.open(path, "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
        return path

    return write


@pytest.fixture
def payload():
    return {
        "format_version": 1,
        "source": {
            "notebook": "notebooks/05_task4_visual_search.ipynb",
            "git_revision": "a758416b7" + "0" * 31,
            "sha256": "1" * 64,
        },
        "cells": {
            "example": [
                {"output_type": "display_data", "data": {
                    "image/png": ["cG5n", "LWJ5dGVz"],
                    "text/plain": ["Recorded ", "figure"],
                }},
                {"output_type": "execute_result", "execution_count": 99, "data": {
                    "text/html": ["<table>", "<tr><td>76.19</td></tr></table>"],
                    "text/plain": "P@10: 76.19",
                }},
            ],
        },
    }


@pytest.fixture
def displayed(monkeypatch):
    calls = []

    def capture(data, **kwargs):
        calls.append((copy.deepcopy(data), kwargs))
        # A display consumer must not be able to change later report runs.
        data.clear()

    monkeypatch.setattr("IPython.display.display", capture)
    return calls


def test_replays_figures_and_tables_with_original_provenance(archive, payload, displayed):
    RecordedNotebookReport(archive(payload)).show("example")

    assert len(displayed) == 3
    label = displayed[0][0]["text/plain"]
    assert "not recomputed" in label
    assert payload["source"]["notebook"] in label
    assert payload["source"]["git_revision"][:9] in label
    assert displayed[1][0] == {
        "image/png": "cG5nLWJ5dGVz", "text/plain": "Recorded figure",
    }
    assert displayed[2][0] == {
        "text/html": "<table><tr><td>76.19</td></tr></table>",
        "text/plain": "P@10: 76.19",
    }
    assert all(kwargs == {"raw": True} for _, kwargs in displayed)


def test_drops_execution_errors_progress_and_unsupported_mime(archive, payload, displayed):
    outputs = payload["cells"]["example"]
    outputs.extend([
        {"output_type": "stream", "name": "stdout", "text": "Saved model"},
        {"output_type": "error", "ename": "RuntimeError", "traceback": ["failure"]},
        {"output_type": "display_data", "data": {"application/javascript": "alert(1)"}},
    ])
    outputs[0]["data"]["application/javascript"] = "alert(2)"
    outputs[0]["data"]["application/vnd.jupyter.widget-view+json"] = {"model_id": "old"}

    RecordedNotebookReport(archive(payload)).show("example")

    assert len(displayed) == 3
    assert all(set(data) <= {"image/png", "text/plain", "text/html"}
               for data, _ in displayed)


def test_image_only_does_not_duplicate_existing_tables(archive, payload, displayed):
    RecordedNotebookReport(archive(payload)).show("example", images_only=True)

    assert len(displayed) == 2
    assert "image/png" in displayed[1][0]
    assert all("text/html" not in data for data, _ in displayed)


def test_rerun_preserves_evidence_and_archive(archive, payload, displayed):
    path = archive(payload)
    original_bytes = path.read_bytes()
    report = RecordedNotebookReport(path)
    report.show("example")
    first_run = copy.deepcopy(displayed)
    displayed.clear()
    report.show("example")

    assert displayed == first_run
    assert path.read_bytes() == original_bytes


def test_missing_archive_fails_with_restore_instruction(tmp_path):
    with pytest.raises(FileNotFoundError, match="Restore this tracked file"):
        RecordedNotebookReport(tmp_path / "missing.json.gz")


def test_missing_cell_fails_explicitly(archive, payload, displayed):
    with pytest.raises(KeyError, match="missing-cell"):
        RecordedNotebookReport(archive(payload)).show("missing-cell")
    assert displayed == []


@pytest.mark.parametrize("images_only", [False, True])
def test_missing_required_static_evidence_does_not_silently_clear_report(
        archive, payload, displayed, images_only):
    payload["cells"]["example"] = (
        [payload["cells"]["example"][1]] if images_only else []
    )
    with pytest.raises(ValueError, match="No static report outputs"):
        RecordedNotebookReport(archive(payload)).show("example", images_only=images_only)
    assert displayed == []


@pytest.mark.parametrize("corruption", [
    "version", "missing_source", "revision_type", "invalid_hash", "cells_type",
    "output_list", "output_type", "display_data", "mime_payload",
])
def test_invalid_archive_fails_explicitly(archive, payload, corruption):
    if corruption == "version":
        payload["format_version"] = 2
    elif corruption == "missing_source":
        payload.pop("source")
    elif corruption == "revision_type":
        payload["source"]["git_revision"] = None
    elif corruption == "invalid_hash":
        payload["source"]["sha256"] = "not-a-checksum"
    elif corruption == "cells_type":
        payload["cells"] = []
    elif corruption == "output_list":
        payload["cells"]["example"] = {}
    elif corruption == "output_type":
        payload["cells"]["example"] = [None]
    elif corruption == "display_data":
        payload["cells"]["example"][0]["data"] = []
    elif corruption == "mime_payload":
        payload["cells"]["example"][0]["data"]["image/png"] = [123]

    with pytest.raises(ValueError):
        RecordedNotebookReport(archive(payload))

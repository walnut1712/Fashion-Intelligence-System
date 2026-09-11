"""Health must expose auxiliary failures as well as the four primary models."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.backend import main


@pytest.mark.parametrize("loaded", [True, False])
def test_health_reports_task3b_availability_and_load_error(monkeypatch, loaded):
    service = SimpleNamespace(health=lambda: {
        "available": True,
        "method": "sfs_category_usage_lift_prior",
        "categories": 50,
        "occasions": 58,
    })
    error = None if loaded else "FileNotFoundError: Task 3B priors missing"
    monkeypatch.setattr(main, "task3b_service", service if loaded else None)
    monkeypatch.setattr(main, "task3b_error", error)

    result = main.health()["models"]["task3b"]

    assert result["loaded"] is loaded
    assert result["available"] is loaded
    assert result["error"] == error
    if loaded:
        assert result["categories"] == 50
        assert result["occasions"] == 58

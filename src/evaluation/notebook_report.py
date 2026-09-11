"""Display archived notebook figures and tables without rerunning experiments.

The archive is independent of the notebook's mutable output cells, so clearing
outputs and saving cannot erase the evidence needed by a subsequent report run.
Only static display formats are replayed; execution, errors and progress streams
are not reproduced.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path
import re


STATIC_MIME_TYPES = frozenset({"image/png", "image/jpeg", "text/html", "text/plain"})


class RecordedNotebookReport:
    def __init__(self, archive_path):
        self.path = Path(archive_path)
        if not self.path.is_file():
            raise FileNotFoundError(
                f"Recorded report archive missing: {self.path}. "
                "Restore this tracked file before running report mode."
            )
        with gzip.open(self.path, "rt", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported recorded report archive format")
        source = payload.get("source")
        cells = payload.get("cells")
        if (not isinstance(source, dict)
                or not isinstance(source.get("notebook"), str)
                or not isinstance(source.get("git_revision"), str)
                or not re.fullmatch(r"[0-9a-f]{40}", source.get("git_revision", ""))
                or not isinstance(source.get("sha256"), str)
                or not re.fullmatch(r"[0-9a-f]{64}", source.get("sha256", ""))
                or not isinstance(cells, dict)):
            raise ValueError("Recorded report archive has invalid provenance or cells")
        self.source = source
        self.cells = {}
        for cell_id, outputs in cells.items():
            if not isinstance(cell_id, str) or not isinstance(outputs, list):
                raise ValueError("Recorded report archive has an invalid cell")
            normalized = []
            for output in outputs:
                if not isinstance(output, dict):
                    raise ValueError(f"Invalid recorded output in cell {cell_id}")
                if output.get("output_type") not in {"display_data", "execute_result"}:
                    continue
                data = output.get("data")
                if not isinstance(data, dict):
                    raise ValueError(f"Invalid recorded display data in cell {cell_id}")
                bundle = {}
                for mime, value in data.items():
                    if mime not in STATIC_MIME_TYPES:
                        continue
                    if isinstance(value, list) and all(isinstance(v, str) for v in value):
                        value = "".join(value)
                    if not isinstance(value, str):
                        raise ValueError(f"Invalid {mime} payload in cell {cell_id}")
                    bundle[mime] = value
                if bundle:
                    normalized.append(bundle)
            self.cells[cell_id] = normalized

    def show(self, cell_id, *, images_only=False):
        """Render static evidence with its original notebook revision identified."""
        if cell_id not in self.cells:
            raise KeyError(f"No recorded report outputs for cell {cell_id}")
        outputs = self.cells[cell_id]
        if images_only:
            outputs = [data for data in outputs
                       if "image/png" in data or "image/jpeg" in data]
        if not outputs:
            raise ValueError(f"No static report outputs available for cell {cell_id}")

        from IPython.display import display

        revision = self.source["git_revision"][:9]
        display({"text/plain": (
            f"Recorded result (not recomputed). Source: {self.source['notebook']} "
            f"at {revision}."
        )}, raw=True)
        for data in outputs:
            display(dict(data), raw=True)

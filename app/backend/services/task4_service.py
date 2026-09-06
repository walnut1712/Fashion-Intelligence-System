from pathlib import Path
import tempfile

import pandas as pd

from src.visual_search.search_engine import PREPROCESS_MODES, SearchEngine


class Task4Service:
    def __init__(self, artifact_dir=None):
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifact_dir = Path(artifact_dir) if artifact_dir else (
            self.project_root / "artifacts" / "task4")
        self.manifest_path = self.artifact_dir / "search_manifest.json"
        if not self.manifest_path.exists():
            raise FileNotFoundError(
                "Task 4 manifest not found: {}".format(self.manifest_path)
            )

        self.engine = SearchEngine.load(self.artifact_dir)
        self.manifest = self.engine.manifest
        self.catalogue_size = len(self.engine.metadata)
        self.embedding_dim = self.engine.index.shape[1]
        self.device = self.engine.device
        self.preprocess_modes = list(PREPROCESS_MODES)

        print("Task 4 loaded:", self.manifest.get("best_method"))
        print("Task 4 catalogue:", self.catalogue_size)
        print("Task 4 embedding dim:", self.embedding_dim)
        print("Task 4 modes:", self.preprocess_modes)

    def _clean_value(self, value):
        if pd.isna(value):
            return None
        if hasattr(value, "item"):
            try:
                return value.item()
            except Exception:
                pass
        return value

    def search(self, image_bytes, k=10, mode="nobg", diversity=0.0):
        if mode not in self.preprocess_modes:
            raise ValueError(
                "Invalid Task 4 preprocessing mode '{}'. Available: {}".format(
                    mode, self.preprocess_modes
                )
            )

        k = max(1, min(24, int(k)))
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = Path(temp_file.name)
        try:
            temp_file.write(image_bytes)
            temp_file.flush()
            temp_file.close()
            results = self.engine.search(
                [temp_path], k=k, mode=mode, diversity=diversity,
                with_diagnostics=True,
            )
        finally:
            if not temp_file.closed:
                temp_file.close()
            temp_path.unlink(missing_ok=True)

        if results is None or len(results) == 0:
            return {"items": [], "diagnostics": {}}

        output = []
        for position, (_, row) in enumerate(results.reset_index(drop=True).iterrows()):
            item_id = self._clean_value(row.get("id"))
            if item_id is not None:
                try:
                    item_id = int(item_id)
                except Exception:
                    item_id = str(item_id)

            similarity = self._clean_value(row.get("similarity"))
            if similarity is not None:
                similarity = round(float(similarity), 4)

            output.append({
                "rank": position + 1,
                "id": item_id,
                "articleType": self._clean_value(row.get("articleType")),
                "subCategory": self._clean_value(row.get("subCategory")),
                "masterCategory": self._clean_value(row.get("masterCategory")),
                "baseColour": self._clean_value(row.get("baseColour")),
                "gender": self._clean_value(row.get("gender")),
                "usage": self._clean_value(row.get("usage")),
                "productDisplayName": self._clean_value(row.get("productDisplayName")),
                "similarity": similarity,
                "image_url": "/api/catalogue/{}/image".format(item_id)
                if item_id is not None else None,
            })

        # Which segmentation tier ran, whether it fell back, and whether the
        # match is worth presenting as confident. All of this already existed in
        # the ingestion layer; the service simply never surfaced it, so a silent
        # fallback to a centre crop looked identical to a clean removal.
        first = results.iloc[0]
        diagnostics = {
            "top1_similarity": self._clean_value(first.get("top1_similarity")),
            "coherence": self._clean_value(first.get("coherence")),
            "confident": bool(first.get("confident", False)),
            "ingest_method": self._clean_value(first.get("ingest_method")),
            "ingest_fell_back": bool(first.get("ingest_fell_back", False)),
        }
        return {"items": output, "diagnostics": diagnostics}

    def resolve_image_path(self, item_id):
        item_id = str(item_id)
        candidates = [
            self.project_root / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train",
            self.project_root / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test",
            self.project_root / "A2_FashionDataset" / "train" / "images_train",
            self.project_root / "A2_FashionDataset" / "test" / "images_test",
            self.project_root / "A2_FashionDataset" / "images_train",
            self.project_root / "A2_FashionDataset" / "FashionDataset" / "images_train",
        ]
        for folder in candidates:
            for extension in (".jpg", ".jpeg", ".png"):
                path = folder / (item_id + extension)
                if path.exists():
                    return path

        metadata = self.engine.metadata
        if "id" in metadata.columns:
            matches = metadata[metadata["id"].astype(str) == item_id]
            if len(matches) > 0 and "image_path" in matches.columns:
                metadata_path = Path(str(matches.iloc[0]["image_path"]))
                if metadata_path.exists():
                    return metadata_path
        return None

    def list_test_sample_ids(self):
        folder = self.project_root / "A2_FashionDataset" / "FashionDataset" / "test" / "images_test"
        if not folder.exists():
            return []
        return sorted(
            int(path.stem)
            for path in folder.glob("*.jpg")
            if path.stem.isdigit()
        )

    def model_card(self):
        """The published-metrics row the frontend shows, straight from the manifest.

        Matches the shape of an ``FI.metrics.tasks[]`` entry in
        ``app/frontend/demo-data.js``. Without this the UI fell back to its
        hardcoded copy, which still advertised the clean baseline's numbers
        (mAP@10 76.7, nDCG@10 89.6) while the service served a different,
        background-augmented encoder.
        """
        manifest = self.manifest or {}
        clean = manifest.get("clean_metrics", {})

        # Prefer the disjoint-bank measurement. ``hard_metrics`` was measured
        # against the background families the encoder trains on, which is a model
        # graded on its own augmentation, and the manifest says as much in its
        # provenance note - yet those were the numbers this card published (52.8
        # where the disjoint measurement is 60.6). Fall back only when no disjoint
        # figure has been recorded.
        hard = manifest.get("hard_metrics_disjoint") or manifest.get("hard_metrics", {})
        hard_is_disjoint = bool(manifest.get("hard_metrics_disjoint"))

        clean_p10 = float(clean.get("P@10", 0.0)) * 100
        clean_colour = float(clean.get("colour@10", 0.0)) * 100
        hard_p10 = float(hard.get("P@10", 0.0)) * 100

        # The reported metrics come from the EVALUATION index, which excludes the
        # held-out products so the queries are unseen. The served index covers the
        # whole catalogue and is larger; quoting the served size here would claim
        # the metrics were measured against items they never searched.
        scored_against = manifest.get("evaluation_catalogue_size",
                                      manifest.get("catalogue_size",
                                                   len(self.engine.index)))
        served = manifest.get("catalogue_size", len(self.engine.index))
        detail = (
            "colour@10 <b>{:.1f}</b> on {:,} held-out queries against a "
            "{:,}-item evaluation index. Serving {:,} items. "
            "Random baseline is <b>5.9%</b>."
        ).format(clean_colour, 2000, scored_against, served)

        note = (
            "On photographs composited onto {} P@10 falls to <b>{:.1f}</b>. The "
            "catalogue is flat-lay product shots, so a real upload is harder than "
            "this headline."
        ).format("backgrounds unseen in training" if hard_is_disjoint
                 else "new backgrounds", hard_p10) if hard else (
            "No out-of-domain benchmark recorded for this encoder."
        )

        return {
            "id": "Task 4",
            "name": "Visual search",
            "headline": round(clean_p10, 1),
            "headlineLabel": "P@10 (same articleType)",
            "detail": detail,
            "flag": "ok" if clean_p10 >= 75 else "warn",
            "flagText": "{} · {}".format(
                manifest.get("best_method", "unknown"),
                manifest.get("provenance", {}).get("trained_by", "unknown notebook"),
            ),
            "note": note,
        }

    def search_regions(self, image_bytes, k=10, mode="nobg"):
        """Per-garment search: propose regions, search each, keep the good ones.

        The encoder produces one vector per image, so a photograph of someone
        wearing a shirt, jeans and shoes is otherwise answered with a single
        guess. Notebook 06 section 11 measured the alternative: 21 of 31 real
        uploads found a closer match from a region than from the whole frame.
        """
        if mode not in self.preprocess_modes:
            raise ValueError(
                "Invalid Task 4 preprocessing mode '{}'. Available: {}".format(
                    mode, self.preprocess_modes
                )
            )

        k = max(1, min(24, int(k)))
        temp_file = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
        temp_path = Path(temp_file.name)
        try:
            temp_file.write(image_bytes)
            temp_file.flush()
            temp_file.close()
            results = self.engine.search_regions(temp_path, k=k)
        finally:
            if not temp_file.closed:
                temp_file.close()
            temp_path.unlink(missing_ok=True)

        if results is None or len(results) == 0:
            return {"regions": [], "segmentation": None}

        regions = []
        for name, group in results.groupby("region", sort=False):
            group = group.reset_index(drop=True)
            items = []
            for position, row in group.iterrows():
                item_id = self._clean_value(row.get("id"))
                if item_id is not None:
                    try:
                        item_id = int(item_id)
                    except Exception:
                        item_id = str(item_id)
                similarity = self._clean_value(row.get("similarity"))
                items.append({
                    "rank": position + 1,
                    "id": item_id,
                    "articleType": self._clean_value(row.get("articleType")),
                    "baseColour": self._clean_value(row.get("baseColour")),
                    "productDisplayName": self._clean_value(row.get("productDisplayName")),
                    "similarity": round(float(similarity), 4)
                    if similarity is not None else None,
                    "image_url": "/api/catalogue/{}/image".format(item_id)
                    if item_id is not None else None,
                })
            regions.append({
                "region": str(name),
                "accepted": bool(group["accepted"].iloc[0]),
                "coherence": self._clean_value(group["coherence"].iloc[0]),
                "top_similarity": round(float(group["similarity"].iloc[0]), 4),
                "items": items,
            })

        # A component that is the whole subject returns the whole image's list
        # again - which happens whenever segmentation finds one connected blob,
        # and only 1 of the 31 real uploads ever split into more. Showing the
        # same twelve results under two headings is noise, so drop the copy and
        # keep the one the reader can interpret.
        whole = next((r for r in regions if r["region"] == "whole"), None)
        if whole is not None:
            signature = [item["id"] for item in whole["items"]]
            regions = [r for r in regions
                       if r["region"] == "whole"
                       or [item["id"] for item in r["items"]] != signature]

        # Accepted regions first, then by how close the match was.
        regions.sort(key=lambda r: (not r["accepted"], -r["top_similarity"]))
        return {
            "regions": regions,
            "segmentation": self._clean_value(results["segmentation"].iloc[0]),
            "accepted_count": sum(1 for r in regions if r["accepted"]),
        }

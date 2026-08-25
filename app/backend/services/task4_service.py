from pathlib import Path
import tempfile

import pandas as pd

from src.visual_search.search_engine import PREPROCESS_MODES, SearchEngine


class Task4Service:
    def __init__(self):
        self.project_root = Path(__file__).resolve().parents[3]
        self.artifact_dir = self.project_root / "artifacts" / "task4"
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

    def search(self, image_bytes, k=10, mode="nobg"):
        if mode not in self.preprocess_modes:
            raise ValueError(
                "Invalid Task 4 preprocessing mode '{}'. Available: {}".format(
                    mode, self.preprocess_modes
                )
            )

        k = max(1, min(20, int(k)))
        with tempfile.NamedTemporaryFile(suffix=".jpg") as temp_file:
            temp_file.write(image_bytes)
            temp_file.flush()
            results = self.engine.search(
                [Path(temp_file.name)], k=k, mode=mode
            )

        if results is None or len(results) == 0:
            return []

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
        return output

    def resolve_image_path(self, item_id):
        item_id = str(item_id)
        candidates = [
            self.project_root / "A2_FashionDataset" / "FashionDataset" / "train" / "images_train",
            self.project_root / "A2_FashionDataset" / "train" / "images_train",
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

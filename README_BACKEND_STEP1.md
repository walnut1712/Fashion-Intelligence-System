# Backend Setup — FastAPI + Tasks 1–4

This backend runs item type, season, gender, usage, and visual search over the
catalogue.

## 1. Copy files into your existing project

Expected project structure:

```text
Fashion-Intelligence-System/
├── app/
│   ├── backend/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   └── services/
│   │       ├── __init__.py
│   │       └── task1_service.py
│   └── frontend/
├── src/
│   └── models/
│       └── item_type_classifier.py   # Task 1 architecture — REQUIRED by the backend
├── artifacts/
│   ├── task1/task1_cnn.pt
│   ├── task2/task2_season_best_pytorch.pth
│   ├── task3/task3_multitask_cnn.pt
│   └── task4/
│       ├── search_manifest.json
│       ├── task4_improved_encoder.pt
│       └── search_index_bg_augmented.npy
└── ...
```

The code automatically expects:

```text
artifacts/task1/task1_cnn.pt
```

`app/backend/services/task1_service.py` does **not** define the Task 1 network
itself — it imports `ItemTypeCNN` from `src/models/item_type_classifier.py`, the
same module `notebooks/02_task1_item_type.ipynb` trains with. Copying `app/`
without `src/` will make Task 1 fail to load. Keeping two copies of the
architecture is what previously left a trained checkpoint that the API could
not load at all, so the duplicate was removed rather than resynchronised.

Verify the checkpoint and the serving path agree before starting the API:

```bash
python -m pytest tests/test_models.py tests/test_prediction.py -q
```

To score a folder of images from the command line instead of over HTTP:

```bash
python predict.py --images A2_FashionDataset/FashionDataset/test/images_test \
                  --out outputs/task1_item_type_predictions.csv --submission
```

## 2. Create/activate your Python environment

Install dependencies:

```bash
pip install -r requirements-backend.txt
```

## 3. Start the API

Run this command from the PROJECT ROOT, not from `app/backend`:

```bash
python -m uvicorn app.backend.main:app --reload
```

Open:

```text
http://127.0.0.1:8000/docs
```

## 4. Check model status

Open:

```text
http://127.0.0.1:8000/api/health
```

The response should show `loaded: true` for Tasks 1–4. The current deployment
normally reports Task 3 as 5 gender classes and 4 usage classes, and Task 4 as
a 32,837-item catalogue with 128-dimensional embeddings.

```json
{"status": "ok", "models": {"task1": {"loaded": true}, "task2": {"loaded": true}, "task3": {"loaded": true}, "task4": {"loaded": true}}
```

If `loaded` is false, read the `error` field. It will show the exact expected checkpoint path.

## 5. Test the API in Swagger

Go to:

```text
http://127.0.0.1:8000/docs
```

Choose one of these endpoints:

```text
POST /api/task1/predict
POST /api/task2/predict
POST /api/task3/predict
POST /api/task4/search
```

Click `Try it out` → upload a JPG/PNG → `Execute`.

For the complete pipeline, use `POST /api/analyze`, upload one image, and set
`k` and `search_mode` in the query parameters.

The response includes:

```json
{
  "filename": "shirt.jpg",
  "predictions": {
    "articleType": {},
    "season": {},
    "gender": {},
    "usage": {}
  }
}
```

Visual search results are in `visual_search.similar_items`. Each result has an
`image_url` such as `/api/catalogue/51808/image` for displaying the catalogue
image.

Useful endpoints:

- `GET /api/health`
- `POST /api/analyze`
- `GET /api/test-samples`
- `GET /api/catalogue/{item_id}/image`

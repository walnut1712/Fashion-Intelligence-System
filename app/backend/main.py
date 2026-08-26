"""Fashion Intelligence System - API backend.

Serves the trained task models to the frontend in app/frontend.

Endpoint contract expected by frontend/app.js:

    POST /api/analyse   (British spelling - this is what the frontend calls)
        form field "image", optional "k"
        -> {latency_ms, predictions: {item_type|season|gender|usage: [{label, p}, ...]},
            results: [...]}
"""

import io
import time
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.backend.services.task1_service import Task1Service
from app.backend.services.task2_service import Task2Service
from app.backend.services.task3_service import Task3Service

PROJECT_ROOT = Path(__file__).resolve().parents[2]

app = FastAPI(title="Fashion Intelligence System API", version="0.3.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

SERVICES = {"task1": None, "task2": None, "task3": None}
ERRORS = {"task1": None, "task2": None, "task3": None}


@app.on_event("startup")
def load_models():
    for name, factory in (("task1", Task1Service), ("task2", Task2Service),
                          ("task3", Task3Service)):
        try:
            SERVICES[name] = factory()
            ERRORS[name] = None
            print("{} model ready".format(name))
        except Exception as error:
            SERVICES[name] = None
            ERRORS[name] = "{}: {}".format(type(error).__name__, error)
            print("{} failed: {}".format(name, ERRORS[name]))


def _read_image(image: UploadFile, data: bytes):
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image")
    if not data:
        raise HTTPException(status_code=400, detail="Empty image")


def _as_ranked(prediction):
    """Normalise a single-label service response into [{label, p}, ...]."""
    if isinstance(prediction, list):
        return prediction
    ranked = prediction.get("top3") or [prediction]
    return [{"label": item["label"], "p": item.get("confidence", item.get("p", 0.0))}
            for item in ranked]


@app.get("/")
def root():
    return {"message": "Fashion Intelligence System backend is running"}


@app.get("/api/health")
def health():
    def describe(name):
        service = SERVICES[name]
        return {
            "loaded": service is not None,
            "classes": getattr(service, "num_classes", None) if service else None,
            "device": str(service.device) if service else None,
            "error": ERRORS[name],
        }

    return {"status": "ok",
            "models": {"task1": describe("task1"), "task2": describe("task2"),
                       "task3": describe("task3"), "task4": {"loaded": False}}}


@app.post("/api/task3/predict")
async def predict_task3(image: UploadFile = File(...)):
    if SERVICES["task3"] is None:
        raise HTTPException(status_code=503, detail=ERRORS["task3"])
    data = await image.read()
    _read_image(image, data)
    try:
        prediction = SERVICES["task3"].predict(data)
    except Exception as error:
        raise HTTPException(status_code=500,
                            detail="{}: {}".format(type(error).__name__, error))
    return {"filename": image.filename, "prediction": prediction}


@app.post("/api/task1/predict")
async def predict_task1(image: UploadFile = File(...)):
    if SERVICES["task1"] is None:
        raise HTTPException(status_code=503, detail=ERRORS["task1"])
    data = await image.read()
    _read_image(image, data)
    try:
        return {"filename": image.filename, "prediction": SERVICES["task1"].predict(data)}
    except Exception as error:
        raise HTTPException(status_code=500,
                            detail="{}: {}".format(type(error).__name__, error))


@app.post("/api/task2/predict")
async def predict_task2(image: UploadFile = File(...)):
    if SERVICES["task2"] is None:
        raise HTTPException(status_code=503, detail=ERRORS["task2"])
    data = await image.read()
    _read_image(image, data)
    try:
        return {"filename": image.filename, "prediction": SERVICES["task2"].predict(data)}
    except Exception as error:
        raise HTTPException(status_code=500,
                            detail="{}: {}".format(type(error).__name__, error))


@app.post("/api/analyse")
async def analyse(image: UploadFile = File(...), k: int = Form(12)):
    """Every available model, in the shape the frontend renders directly."""
    data = await image.read()
    _read_image(image, data)

    started = time.perf_counter()
    predictions, failures = {}, {}

    if SERVICES["task1"] is not None:
        try:
            predictions["item_type"] = _as_ranked(SERVICES["task1"].predict(data))
        except Exception as error:
            failures["item_type"] = str(error)

    if SERVICES["task2"] is not None:
        try:
            predictions["season"] = _as_ranked(SERVICES["task2"].predict(data))
        except Exception as error:
            failures["season"] = str(error)

    if SERVICES["task3"] is not None:
        try:
            task3 = SERVICES["task3"].predict(data)
            predictions["gender"] = task3["gender"]
            predictions["usage"] = task3["usage"]
        except Exception as error:
            failures["gender"] = failures["usage"] = str(error)

    if not predictions:
        raise HTTPException(status_code=503,
                            detail="No models are loaded. See /api/health.")

    return {
        "filename": image.filename,
        "latency_ms": int((time.perf_counter() - started) * 1000),
        "predictions": predictions,
        "results": [],                  # Task 4 visual search is not wired up yet
        "unavailable": failures or None,
    }


# American spelling kept as an alias so older clients keep working.
@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...), k: int = Form(12)):
    return await analyse(image=image, k=k)


@app.get("/api/image/{item_id}")
def catalogue_image(item_id: str):
    """Thumbnail for the 'Use a sample item' button."""
    for folder in ("train/images_train", "test/images_test"):
        path = (PROJECT_ROOT / "A2_FashionDataset" / "FashionDataset" / folder
                / "{}.jpg".format(item_id))
        if path.exists():
            return FileResponse(path, media_type="image/jpeg")
    raise HTTPException(status_code=404, detail="No image for id {}".format(item_id))

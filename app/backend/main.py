from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from app.backend.services.task1_service import Task1Service
from app.backend.services.task2_service import Task2Service
from app.backend.services.task3_service import Task3Service
from app.backend.services.task4_service import Task4Service


app = FastAPI(title="Fashion Intelligence System API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

task1_service = None
task1_error = None
task2_service = None
task2_error = None
task3_service = None
task3_error = None
task4_service = None
task4_error = None




@app.on_event("startup")
def load_models():
    global task1_service, task1_error, task2_service, task2_error
    global task3_service, task3_error
    global task4_service, task4_error
    try:
        task1_service = Task1Service()
        task1_error = None
        print("Task 1 model ready")
    except Exception as error:
        task1_service = None
        task1_error = "{}: {}".format(type(error).__name__, error)

        print("Task 1 failed:", task1_error)

    try:
        task2_service = Task2Service()
        task2_error = None
        print("Task 2 model ready")
    except Exception as error:
        task2_service = None
        task2_error = "{}: {}".format(type(error).__name__, error)
        print("Task 2 failed:", task2_error)

    try:
        task3_service = Task3Service()
        task3_error = None
        print("Task 3 model ready")
    except Exception as error:
        task3_service = None
        task3_error = "{}: {}".format(type(error).__name__, error)
        print("Task 3 failed:", task3_error)

    try:
        task4_service = Task4Service()
        task4_error = None
        print("Task 4 search engine ready")
    except Exception as error:
        task4_service = None
        task4_error = "{}: {}".format(type(error).__name__, error)
        print("Task 4 failed:", task4_error)


@app.get("/")

def root():
    return {"message": "Fashion Intelligence System backend is running"}


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "models": {
            "task1": {
                "loaded": task1_service is not None,
                "classes": task1_service.num_classes if task1_service else None,
                "device": str(task1_service.device) if task1_service else None,
                "error": task1_error,
            },
            "task2": {
                "loaded": task2_service is not None,
                "classes": task2_service.num_classes if task2_service else None,
                "device": str(task2_service.device) if task2_service else None,
                "error": task2_error,
            },
            "task3": {
                "loaded": task3_service is not None,
                "classes": task3_service.num_classes if task3_service else None,
                "device": str(task3_service.device) if task3_service else None,
                "error": task3_error,
            },
            "task4": {
                "loaded": task4_service is not None,
                "catalogue_size": task4_service.catalogue_size if task4_service else None,
                "embedding_dim": task4_service.embedding_dim if task4_service else None,
                "method": task4_service.manifest.get("best_method") if task4_service else None,
                "device": str(task4_service.device) if task4_service else None,
                "error": task4_error,
            },
        },
    }


@app.post("/api/task1/predict")
async def predict_task1(image: UploadFile = File(...)):
    if task1_service is None:
        raise HTTPException(status_code=503, detail=task1_error)
    if image.content_type and not image.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Please upload an image")
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")
    try:

        prediction = task1_service.predict(image_bytes)
    except Exception as error:
        raise HTTPException(status_code=500, detail="{}: {}".format(type(error).__name__, error))
    return {"filename": image.filename, "prediction": prediction}


@app.post("/api/task2/predict")
async def predict_task2(image: UploadFile = File(...)):
    if task2_service is None:
        raise HTTPException(status_code=503, detail=task2_error)
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")
    try:
        prediction = task2_service.predict(image_bytes)
    except Exception as error:
        raise HTTPException(status_code=500, detail="{}: {}".format(type(error).__name__, error))
    return {"filename": image.filename, "prediction": prediction}


@app.post("/api/task3/predict")
async def predict_task3(image: UploadFile = File(...)):
    if task3_service is None:
        raise HTTPException(status_code=503, detail=task3_error)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        prediction = task3_service.predict(image_bytes)
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(type(error).__name__, error),
        )
    return {"filename": image.filename, "prediction": prediction}


@app.post("/api/task4/search")
async def search_task4(
    image: UploadFile = File(...),
    k: int = 10,
    mode: str = "nobg",
):
    if task4_service is None:
        raise HTTPException(status_code=503, detail=task4_error)

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        results = task4_service.search(image_bytes, k=k, mode=mode)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(type(error).__name__, error),
        )

    return {
        "filename": image.filename,
        "k": max(1, min(20, int(k))),
        "mode": mode,
        "method": task4_service.manifest.get("best_method"),
        "results": results,
    }


@app.get("/api/catalogue/{item_id}/image")
def catalogue_image(item_id: str):
    if task4_service is None:
        raise HTTPException(status_code=503, detail=task4_error)

    image_path = task4_service.resolve_image_path(item_id)
    if image_path is None:
        raise HTTPException(status_code=404, detail="Catalogue image not found")
    return FileResponse(str(image_path))


@app.get("/api/test-samples")
def test_samples():
    if task4_service is None:
        raise HTTPException(status_code=503, detail=task4_error)
    return {"ids": task4_service.list_test_sample_ids()}


@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    k: int = 10,
    search_mode: str = "nobg",
):
    if task1_service is None:
        raise HTTPException(
            status_code=503,
            detail="Task 1 unavailable: {}".format(task1_error),
        )
    if task2_service is None:
        raise HTTPException(
            status_code=503,
            detail="Task 2 unavailable: {}".format(task2_error),
        )
    if task3_service is None:
        raise HTTPException(
            status_code=503,
            detail="Task 3 unavailable: {}".format(task3_error),
        )
    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail="Task 4 unavailable: {}".format(task4_error),
        )

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        article_type = task1_service.predict(image_bytes)
        season = task2_service.predict(image_bytes)
        task3_result = task3_service.predict(image_bytes)
        similar_items = task4_service.search(
            image_bytes,
            k=k,
            mode=search_mode,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(type(error).__name__, error),
        )

    return {
        "filename": image.filename,
        "predictions": {
            "articleType": article_type,
            "season": season,
            "gender": task3_result["gender"],
            "usage": task3_result["usage"],
        },
        "visual_search": {
            "method": task4_service.manifest.get("best_method"),
            "mode": search_mode,
            "k": max(1, min(20, int(k))),
            "similar_items": similar_items,
        },
        "backend_stage": "all_tasks_connected",
    }
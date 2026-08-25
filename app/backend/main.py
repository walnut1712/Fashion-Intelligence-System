from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.backend.services.task1_service import Task1Service
from app.backend.services.task2_service import Task2Service
from app.backend.services.task3_service import Task3Service


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




@app.on_event("startup")
def load_models():
    global task1_service, task1_error, task2_service, task2_error
    global task3_service, task3_error
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
            "task4": {"loaded": False},
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


@app.post("/api/analyze")
async def analyze(image: UploadFile = File(...)):
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

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty image")

    try:
        article_type = task1_service.predict(image_bytes)
        season = task2_service.predict(image_bytes)
        task3_result = task3_service.predict(image_bytes)
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
        "similar_items": [],
        "backend_stage": "task1_task2_task3_connected",
    }
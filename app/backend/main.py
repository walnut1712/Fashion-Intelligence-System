import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.services.task1_service import Task1Service
from app.backend.services.task2_service import Task2Service
from app.backend.services.task3_service import Task3Service
from app.backend.services.task4_service import Task4Service


# ============================================================
# GLOBAL SERVICES
# ============================================================

task1_service = None
task1_error = None

task2_service = None
task2_error = None

task3_service = None
task3_error = None

task4_service = None
task4_error = None


# ============================================================
# MODEL LOADING
# ============================================================

def load_models():
    """
    Load each task independently so one failed checkpoint
    does not bring down the whole API.
    """

    global task1_service, task1_error
    global task2_service, task2_error
    global task3_service, task3_error
    global task4_service, task4_error

    # Task 1
    try:
        task1_service = Task1Service()
        task1_error = None
        print("Task 1 model ready")

    except Exception as error:
        task1_service = None
        task1_error = "{}: {}".format(
            type(error).__name__,
            error,
        )
        print("Task 1 failed:", task1_error)

    # Task 2
    try:
        task2_service = Task2Service()
        task2_error = None
        print("Task 2 model ready")

    except Exception as error:
        task2_service = None
        task2_error = "{}: {}".format(
            type(error).__name__,
            error,
        )
        print("Task 2 failed:", task2_error)

    # Task 3
    try:
        task3_service = Task3Service()
        task3_error = None
        print("Task 3 model ready")

    except Exception as error:
        task3_service = None
        task3_error = "{}: {}".format(
            type(error).__name__,
            error,
        )
        print("Task 3 failed:", task3_error)

    # Task 4
    try:
        task4_service = Task4Service()
        task4_error = None
        print("Task 4 search engine ready")

    except Exception as error:
        task4_service = None
        task4_error = "{}: {}".format(
            type(error).__name__,
            error,
        )
        print("Task 4 failed:", task4_error)


# ============================================================
# APP LIFESPAN
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Fashion Intelligence System API",
    version="0.3.0",
    lifespan=lifespan,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LIVE META
# ============================================================

def _live_meta():
    meta = {}

    if task1_service is not None:
        meta["itemType"] = task1_service.class_names

    return meta


# ============================================================
# LIVE METRICS
# ============================================================

def _live_metrics():
    tasks = []

    if task1_service is not None:
        tasks.append(
            task1_service.model_card()
        )

    if not tasks:
        return None

    return {
        "source":
            "artifacts/task{1,2,3,4}/*.json "
            "(live from loaded checkpoints)",
        "tasks": tasks,
    }


# ============================================================
# HEALTH
# ============================================================

@app.get("/api/health")
def health():

    return {
        "status": "ok",

        "meta": _live_meta(),

        "metrics": _live_metrics(),

        "models": {

            "task1": {
                "loaded":
                    task1_service is not None,

                "classes":
                    task1_service.num_classes
                    if task1_service
                    else None,

                "device":
                    str(task1_service.device)
                    if task1_service
                    else None,

                "error":
                    task1_error,
            },

            "task2": {
                "loaded":
                    task2_service is not None,

                "classes":
                    task2_service.num_classes
                    if task2_service
                    else None,

                "device":
                    str(task2_service.device)
                    if task2_service
                    else None,

                "error":
                    task2_error,
            },

            "task3": {
                "loaded":
                    task3_service is not None,

                "classes":
                    task3_service.num_classes
                    if task3_service
                    else None,

                "device":
                    str(task3_service.device)
                    if task3_service
                    else None,

                "error":
                    task3_error,
            },

            "task4": {
                "loaded":
                    task4_service is not None,

                "catalogue_size":
                    task4_service.catalogue_size
                    if task4_service
                    else None,

                "embedding_dim":
                    task4_service.embedding_dim
                    if task4_service
                    else None,

                "method":
                    task4_service.manifest.get(
                        "best_method"
                    )
                    if task4_service
                    else None,

                "device":
                    str(task4_service.device)
                    if task4_service
                    else None,

                "error":
                    task4_error,
            },
        },
    }


# ============================================================
# TASK 1 ENDPOINT
# ============================================================

@app.post("/api/task1/predict")
async def predict_task1(
    image: UploadFile = File(...),
    ingest: str = None,
):

    if task1_service is None:
        raise HTTPException(
            status_code=503,
            detail=task1_error,
        )

    if (
        image.content_type
        and not image.content_type.startswith("image/")
    ):
        raise HTTPException(
            status_code=415,
            detail="Please upload an image",
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    try:
        prediction = task1_service.predict(
            image_bytes,
            ingest=ingest,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(
                type(error).__name__,
                error,
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
    }


# ============================================================
# TASK 2 ENDPOINT
# ============================================================

@app.post("/api/task2/predict")
async def predict_task2(
    image: UploadFile = File(...),
):

    if task2_service is None:
        raise HTTPException(
            status_code=503,
            detail=task2_error,
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    try:

        # ----------------------------------------------------
        # Task 1 context
        # ----------------------------------------------------

        article_type = None
        article_type_confidence = None

        article_family = None
        article_family_confidence = None

        if task1_service is not None:

            task1_result = task1_service.predict(
                image_bytes
            )

            if isinstance(
                task1_result,
                dict,
            ):

                # Article type
                article_type = (
                    task1_result.get(
                        "label"
                    )
                )

                article_type_confidence = (
                    task1_result.get(
                        "confidence"
                    )
                )

                # Family / subCategory
                family_result = (
                    task1_result.get(
                        "family"
                    )
                    or {}
                )

                if isinstance(
                    family_result,
                    dict,
                ):

                    article_family = (
                        family_result.get(
                            "label"
                        )
                    )

                    article_family_confidence = (
                        family_result.get(
                            "confidence"
                        )
                    )

        # ----------------------------------------------------
        # Task 2 prediction
        # ----------------------------------------------------

        prediction = task2_service.predict(

            image_bytes,

            article_type=
                article_type,

            article_type_confidence=
                article_type_confidence,

            article_family=
                article_family,

            article_family_confidence=
                article_family_confidence,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(
                type(error).__name__,
                error,
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
    }


# ============================================================
# TASK 3 ENDPOINT
# ============================================================

@app.post("/api/task3/predict")
async def predict_task3(
    image: UploadFile = File(...),
):

    if task3_service is None:
        raise HTTPException(
            status_code=503,
            detail=task3_error,
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    try:
        prediction = task3_service.predict(
            image_bytes
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(
                type(error).__name__,
                error,
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
    }


# ============================================================
# TASK 4 SEARCH
# ============================================================

@app.post("/api/task4/search")
async def search_task4(

    image: UploadFile = File(...),

    k: int = 10,

    mode: str = "nobg",
):

    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=task4_error,
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    try:
        results = task4_service.search(
            image_bytes,
            k=k,
            mode=mode,
        )

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(
                type(error).__name__,
                error,
            ),
        )

    return {
        "filename":
            image.filename,

        "k":
            max(
                1,
                min(
                    20,
                    int(k),
                ),
            ),

        "mode":
            mode,

        "method":
            task4_service.manifest.get(
                "best_method"
            ),

        "results":
            results,
    }


# ============================================================
# CATALOGUE IMAGE
# ============================================================

@app.get(
    "/api/catalogue/{item_id}/image"
)
def catalogue_image(
    item_id: str,
):

    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=task4_error,
        )

    image_path = (
        task4_service.resolve_image_path(
            item_id
        )
    )

    if image_path is None:
        raise HTTPException(
            status_code=404,
            detail="Catalogue image not found",
        )

    return FileResponse(
        str(image_path)
    )


# ============================================================
# TEST SAMPLES
# ============================================================

@app.get("/api/test-samples")
def test_samples():

    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=task4_error,
        )

    return {
        "ids":
            task4_service.list_test_sample_ids()
    }


# ============================================================
# FULL ANALYSIS ENDPOINT
# ============================================================

@app.post("/api/analyze")
async def analyze(

    image: UploadFile = File(...),

    k: int = 10,

    search_mode: str = "nobg",

    ingest: str = None,
):

    # --------------------------------------------------------
    # Ensure all models are available
    # --------------------------------------------------------

    if task1_service is None:
        raise HTTPException(
            status_code=503,
            detail=
                "Task 1 unavailable: {}".format(
                    task1_error
                ),
        )

    if task2_service is None:
        raise HTTPException(
            status_code=503,
            detail=
                "Task 2 unavailable: {}".format(
                    task2_error
                ),
        )

    if task3_service is None:
        raise HTTPException(
            status_code=503,
            detail=
                "Task 3 unavailable: {}".format(
                    task3_error
                ),
        )

    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=
                "Task 4 unavailable: {}".format(
                    task4_error
                ),
        )

    # --------------------------------------------------------
    # Read uploaded image
    # --------------------------------------------------------

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    started = time.perf_counter()

    try:

        # ====================================================
        # TASK 1
        # ====================================================

        article_type = (
            task1_service.predict(
                image_bytes,
                ingest=ingest,
            )
        )

        # ====================================================
        # TASK 1 ARTICLE TYPE
        # ====================================================

        article_type_label = None
        article_type_confidence = None

        if isinstance(
            article_type,
            dict,
        ):

            article_type_label = (
                article_type.get(
                    "label"
                )
            )

            article_type_confidence = (
                article_type.get(
                    "confidence"
                )
            )

        # ====================================================
        # TASK 1 FAMILY / CATEGORY
        # ====================================================

        article_family_label = None
        article_family_confidence = None

        if isinstance(
            article_type,
            dict,
        ):

            article_family_result = (
                article_type.get(
                    "family"
                )
                or {}
            )

            if isinstance(
                article_family_result,
                dict,
            ):

                article_family_label = (
                    article_family_result.get(
                        "label"
                    )
                )

                article_family_confidence = (
                    article_family_result.get(
                        "confidence"
                    )
                )

        # ====================================================
        # TASK 2
        #
        # Official CNN:
        # Fall / Spring / Summer / Winter
        #
        # Semantic display layer can output:
        #
        # All Season
        # Spring / Summer
        # Summer / Fall
        # Fall / Winter
        # Winter / Spring
        # ====================================================

        season = (
            task2_service.predict(

                image_bytes,

                article_type=
                    article_type_label,

                article_type_confidence=
                    article_type_confidence,

                article_family=
                    article_family_label,

                article_family_confidence=
                    article_family_confidence,
            )
        )

        # ====================================================
        # TASK 3
        # ====================================================

        task3_result = (
            task3_service.predict(
                image_bytes
            )
        )

        # ====================================================
        # TASK 4
        # ====================================================

        similar_items = (
            task4_service.search(
                image_bytes,
                k=k,
                mode=search_mode,
            )
        )

    except ValueError as error:

        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:

        raise HTTPException(
            status_code=500,
            detail="{}: {}".format(
                type(error).__name__,
                error,
            ),
        )

    # ========================================================
    # RESPONSE
    # ========================================================

    return {

        "filename":
            image.filename,

        "latency_ms":
            round(
                (
                    time.perf_counter()
                    - started
                )
                * 1000,
                1,
            ),

        "predictions": {

            "articleType":
                article_type,

            "season":
                season,

            "gender":
                task3_result[
                    "gender"
                ],

            "usage":
                task3_result[
                    "usage"
                ],
        },

        "visual_search": {

            "method":
                task4_service
                .manifest
                .get(
                    "best_method"
                ),

            "mode":
                search_mode,

            "k":
                max(
                    1,
                    min(
                        20,
                        int(k),
                    ),
                ),

            "similar_items":
                similar_items,
        },

        "backend_stage":
            "task2_semantic_policy_v2",
    }


# ============================================================
# FRONTEND
# ============================================================

FRONTEND = (
    Path(__file__)
    .resolve()
    .parents[1]
    / "frontend"
)


app.mount(
    "/",
    StaticFiles(
        directory=FRONTEND,
        html=True,
    ),
    name="frontend",
)
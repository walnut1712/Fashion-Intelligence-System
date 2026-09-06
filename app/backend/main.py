import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.backend.services.task1_service import Task1Service
from app.backend.services.task2_service import Task2Service
from app.backend.services.task2b_service import Task2BService
from app.backend.services.task3_service import Task3Service
from app.backend.services.task3b_service import Task3BService
from app.backend.services.task4_service import Task4Service


# ============================================================
# Services
# ============================================================

task1_service = None
task1_error = None

task2_service = None
task2_error = None

task2b_service = None
task2b_error = None

task3_service = None
task3_error = None

task3b_service = None
task3b_error = None

task4_service = None
task4_error = None


def load_models():
    """
    Load each task independently.

    Task2A:
        official catalogue-season CNN

    Task2B:
        auxiliary SFS-based suitable-season recommendation

    A Task2B failure must NOT take the official Task2A pipeline down.
    """
    global task1_service, task1_error
    global task2_service, task2_error
    global task2b_service, task2b_error
    global task3_service, task3_error
    global task3b_service, task3b_error
    global task4_service, task4_error

    # --------------------------------------------------------
    # Task 1
    # --------------------------------------------------------
    try:
        task1_service = Task1Service()
        task1_error = None
        print("Task 1 model ready")

    except Exception as error:
        task1_service = None
        task1_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 1 failed:",
            task1_error,
        )

    # --------------------------------------------------------
    # Task 2A
    # --------------------------------------------------------
    try:
        task2_service = Task2Service()
        task2_error = None
        print("Task 2A catalogue-season model ready")

    except Exception as error:
        task2_service = None
        task2_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 2A failed:",
            task2_error,
        )

    # --------------------------------------------------------
    # Task 2B
    # --------------------------------------------------------
    try:
        task2b_service = Task2BService()
        task2b_error = None
        print("Task 2B SFS recommendation model ready")

    except Exception as error:
        task2b_service = None
        task2b_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 2B failed:",
            task2b_error,
        )

    # --------------------------------------------------------
    # Task 3
    # --------------------------------------------------------
    try:
        task3_service = Task3Service()
        task3_error = None
        print("Task 3 model ready")

    except Exception as error:
        task3_service = None
        task3_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 3 failed:",
            task3_error,
        )

    # --------------------------------------------------------
    # Task 3B - auxiliary occasion recommendation
    # --------------------------------------------------------
    try:
        task3b_service = Task3BService()
        task3b_error = None
        print("Task 3B SFS occasion recommendation ready")

    except Exception as error:
        task3b_service = None
        task3b_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 3B failed:",
            task3b_error,
        )

    # --------------------------------------------------------
    # Task 4
    # --------------------------------------------------------
    try:
        task4_service = Task4Service()
        task4_error = None
        print("Task 4 search engine ready")

    except Exception as error:
        task4_service = None
        task4_error = (
            "{}: {}".format(
                type(error).__name__,
                error,
            )
        )
        print(
            "Task 4 failed:",
            task4_error,
        )


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_models()
    yield


# ============================================================
# FastAPI
# ============================================================

app = FastAPI(
    title="Fashion Intelligence System API",
    version="0.3.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# Metadata
# ============================================================

def _live_meta():
    """
    Label vocabularies that the backend can verify.
    """
    meta = {}

    if task1_service is not None:
        meta["itemType"] = (
            task1_service.class_names
        )

    return meta


def _live_metrics():
    """
    Existing official model-card metrics.

    Task2B is deliberately NOT mixed into the official
    assignment Task2 metric row because it is an auxiliary
    external-data recommendation model.
    """
    tasks = []

    if task1_service is not None:
        tasks.append(
            task1_service.model_card()
        )

    if task4_service is not None:
        tasks.append(
            task4_service.model_card()
        )

    if not tasks:
        return None

    return {
        "source": (
            "artifacts/task{1,2,3,4}/*.json "
            "(live from loaded checkpoints)"
        ),
        "tasks": tasks,
    }


# ============================================================
# Health
# ============================================================

@app.get("/api/health")
def health():
    if task2b_service is not None:
        task2b_health = (
            task2b_service.health_info()
        )
        task2b_health["error"] = None

    else:
        task2b_health = {
            "loaded": False,
            "method": "hard_top1_sfs",
            "classes": None,
            "num_classes": None,
            "sfs_categories": None,
            "task1_classes": None,
            "supported_task1_classes": None,
            "supported_fraction": None,
            "device": "cpu",
            "error": task2b_error,
        }

    return {
        "status": "ok",
        "meta": _live_meta(),
        "metrics": _live_metrics(),
        "models": {
            "task1": {
                "loaded":
                    task1_service
                    is not None,
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
                    task2_service
                    is not None,
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
                "role":
                    "official_catalogue_season",
            },

            "task2b": task2b_health,

            "task3": {
                "loaded":
                    task3_service
                    is not None,
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
                    task4_service
                    is not None,
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
# Task 1
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
        and not image.content_type.startswith(
            "image/"
        )
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
        prediction = (
            task1_service.predict(
                image_bytes,
                ingest=ingest,
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
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
    }


# ============================================================
# Task 2A
# ============================================================

@app.post("/api/task2/predict")
async def predict_task2(
    image: UploadFile = File(...),
):
    """
    Official Task2A catalogue-season classifier.
    """
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
        prediction = (
            task2_service.predict(
                image_bytes
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
        "role": "official_catalogue_season",
    }


# ============================================================
# Task 2B
# ============================================================

@app.get("/api/task2b/predict")
def predict_task2b(
    article_type: str,
):
    """
    Direct Task2B diagnostic endpoint.

    Example:
        /api/task2b/predict?article_type=Sandals

    This endpoint does not inspect an image.
    It receives the Task1 articleType directly.
    """
    if task2b_service is None:
        raise HTTPException(
            status_code=503,
            detail=task2b_error,
        )

    try:
        prediction = (
            task2b_service.predict(
                article_type
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
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "article_type": article_type,
        "prediction": prediction,
        "role":
            "auxiliary_suitable_season_recommendation",
    }


# ============================================================
# Task 3
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
        prediction = (
            task3_service.predict(
                image_bytes
            )
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "filename": image.filename,
        "prediction": prediction,
    }


# ============================================================
# Task 4
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
        search_result = task4_service.search(
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
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "filename": image.filename,
        "k": max(
            1,
            min(
                24,
                int(k),
            ),
        ),
        "mode": mode,
        "method":
            task4_service.manifest.get(
                "best_method"
            ),
        "results": search_result["items"],
        "diagnostics": search_result["diagnostics"],
    }


@app.post(
    "/api/task4/regions"
)
async def search_task4_regions(
    image: UploadFile = File(...),
    k: int = 6,
    mode: str = "nobg",
):
    """Per-garment retrieval for a photo containing more than one item.

    The encoder produces one vector per image, so a whole-frame query on an
    outfit averages into a vector describing none of its garments. This returns
    one group per proposed region, accepted ones first.
    """
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
        result = task4_service.search_regions(
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
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    return {
        "filename": image.filename,
        "k": k,
        "mode": mode,
        "method":
            task4_service.manifest.get(
                "best_method"
            ),
        **result,
    }


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
            detail=(
                "Catalogue image not found"
            ),
        )

    return FileResponse(
        str(image_path)
    )


@app.get("/api/test-samples")
def test_samples():
    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=task4_error,
        )

    return {
        "ids":
            task4_service
            .list_test_sample_ids()
    }


# ============================================================
# Combined analysis
# ============================================================

@app.post("/api/analyze")
async def analyze(
    image: UploadFile = File(...),
    k: int = 10,
    search_mode: str = "nobg",
    ingest: str = None,
):
    """
    Run all official tasks and, when available, Task2B.

    Task2A:
        image -> catalogue season

    Task2B:
        Task1 top-1 articleType
            -> SFS taxonomy
            -> suitable-season recommendation

    Task2B is auxiliary and does not replace Task2A.
    """

    # Official tasks remain required.
    if task1_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Task 1 unavailable: {}"
                .format(task1_error)
            ),
        )

    if task2_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Task 2 unavailable: {}"
                .format(task2_error)
            ),
        )

    if task3_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Task 3 unavailable: {}"
                .format(task3_error)
            ),
        )

    if task4_service is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Task 4 unavailable: {}"
                .format(task4_error)
            ),
        )

    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(
            status_code=400,
            detail="Empty image",
        )

    started = time.perf_counter()

    try:
        # ----------------------------------------------------
        # Task 1 first
        # ----------------------------------------------------
        article_type = (
            task1_service.predict(
                image_bytes,
                ingest=ingest,
            )
        )

        # ----------------------------------------------------
        # Official Task 2A remains unchanged
        # ----------------------------------------------------
        season = (
            task2_service.predict(
                image_bytes
            )
        )

        # ----------------------------------------------------
        # Task 3
        # ----------------------------------------------------
        task3_result = (
            task3_service.predict(
                image_bytes
            )
        )

        # ----------------------------------------------------
        # Task 4
        # ----------------------------------------------------
        # search() returns {items, diagnostics}: the confidence verdict travels
        # with the results rather than being recomputed or dropped.
        task4_result = (
            task4_service.search(
                image_bytes,
                k=k,
                mode=search_mode,
            )
        )
        similar_items = task4_result["items"]

    except ValueError as error:
        raise HTTPException(
            status_code=400,
            detail=str(error),
        )

    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail=(
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            ),
        )

    # --------------------------------------------------------
    # Auxiliary Task 2B
    #
    # It is intentionally isolated from the official task
    # execution above. If Task2B fails, the official four-task
    # response still succeeds.
    # --------------------------------------------------------

    season_recommendation = None
    season_recommendation_error = None

    if task2b_service is not None:
        try:
            season_recommendation = (
                task2b_service
                .predict_from_task1(
                    article_type
                )
            )

        except Exception as error:
            season_recommendation_error = (
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            )

    else:
        season_recommendation_error = (
            task2b_error
            or "Task2B service unavailable"
        )

    # --------------------------------------------------------
    # Task 3B - auxiliary fine-grained occasion recommendation
    # --------------------------------------------------------
    usage_recommendation = None
    usage_recommendation_error = None

    if task3b_service is not None:
        try:
            # Task1 production response may be either a ranked list
            # or a {label, confidence, top3} dictionary.
            if isinstance(article_type, list):
                article_type_label = (
                    article_type[0].get("label")
                    if article_type
                    else None
                )

            elif isinstance(article_type, dict):
                article_type_label = article_type.get("label")

                if not article_type_label:
                    ranked = article_type.get("top3") or []
                    article_type_label = (
                        ranked[0].get("label")
                        if ranked
                        else None
                    )

            else:
                article_type_label = str(article_type)

            usage_value = task3_result["usage"]

            if isinstance(usage_value, list):
                usage_label = (
                    usage_value[0].get("label")
                    if usage_value
                    else None
                )

            elif isinstance(usage_value, dict):
                usage_label = usage_value.get("label")

                if not usage_label:
                    ranked = usage_value.get("top3") or []
                    usage_label = (
                        ranked[0].get("label")
                        if ranked
                        else None
                    )

            else:
                usage_label = str(usage_value)

            if article_type_label and usage_label:
                usage_recommendation = (
                    task3b_service.recommend(
                        article_type_label,
                        usage_label,
                    )
                )

        except Exception as error:
            usage_recommendation_error = (
                "{}: {}".format(
                    type(error).__name__,
                    error,
                )
            )

    else:
        usage_recommendation_error = (
            task3b_error
            or "Task3B service unavailable"
        )

    response = {
        "filename": image.filename,

        "latency_ms": round(
            (
                time.perf_counter()
                - started
            )
            * 1000,
            1,
        ),

        "predictions": {
            # Official assignment outputs
            "articleType":
                article_type,

            "season":
                season,

            "gender":
                task3_result["gender"],

            "usage":
                task3_result["usage"],

            # Auxiliary external-data layer
            "season_recommendation":
                season_recommendation,

            "usage_recommendation":
                usage_recommendation,
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

            "k": max(
                1,
                min(
                    24,
                    int(k),
                ),
            ),

            "similar_items":
                similar_items,

            "diagnostics":
                task4_result["diagnostics"],
        },

        "backend_stage":
            "task2b_task3b_connected",
    }

    if (
        season_recommendation_error
        is not None
    ):
        response[
            "season_recommendation_error"
        ] = season_recommendation_error

    if usage_recommendation_error is not None:
        response[
            "usage_recommendation_error"
        ] = usage_recommendation_error

    return response


# ============================================================
# Frontend
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
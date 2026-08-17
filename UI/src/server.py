import logging
import asyncio
import uuid
import os
import json
import time
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

from src.models.schemas import CheckRequest
from src.pipelines.router import dispatch, classify
from src.db.database import create_db_and_tables, log_check

logger = logging.getLogger(__name__)

app = FastAPI(title="Satya API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

active_checks: dict = {}


@app.on_event("startup")
async def startup_event():
    create_db_and_tables()
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("frontend/css", exist_ok=True)
    os.makedirs("frontend/js", exist_ok=True)


# Mount frontend static files
try:
    app.mount("/css", StaticFiles(directory="frontend/css"), name="css")
    app.mount("/js", StaticFiles(directory="frontend/js"), name="js")
except RuntimeError:
    pass


@app.get("/")
async def read_index():
    return FileResponse("frontend/index.html")


@app.get("/api/health")
async def health_check():
    return {"status": "ok"}


async def run_analysis(check_id: str, request: CheckRequest, queue: asyncio.Queue):
    start_time = time.time()
    try:
        async def progress_cb(step: str, status: str, message: str):
            """Send progress updates as plain dicts (not Pydantic models) for simplicity."""
            await queue.put({
                "event": "progress",
                "data": json.dumps({"step": step, "status": status, "message": message})
            })

        evidence, verdict_card = await dispatch(request, progress_cb)

        latency_ms = int((time.time() - start_time) * 1000)
        log_check(check_id, request.message_type.value, verdict_card.verdict.value, verdict_card.confidence, latency_ms)

        await queue.put({"event": "verdict", "data": verdict_card.model_dump_json()})
    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        await queue.put({"event": "error", "data": json.dumps({"error": str(e)})})
    finally:
        await queue.put({"event": "done", "data": ""})


@app.post("/api/check")
async def create_check(
    text: str = Form(None),
    image: UploadFile = File(None),
    audio: UploadFile = File(None)
):
    check_id = str(uuid.uuid4())

    image_path = None
    if image:
        image_path = f"uploads/{check_id}_{image.filename}"
        with open(image_path, "wb") as f:
            f.write(await image.read())

    audio_path = None
    if audio:
        audio_path = f"uploads/{check_id}_{audio.filename}"
        with open(audio_path, "wb") as f:
            f.write(await audio.read())

    request = CheckRequest(
        id=check_id,
        text=text,
        image_path=image_path,
        audio_path=audio_path,
        message_type=classify(text, image_path, audio_path)
    )

    queue: asyncio.Queue = asyncio.Queue()
    task = asyncio.create_task(run_analysis(check_id, request, queue))

    active_checks[check_id] = {"queue": queue, "task": task}

    return {"id": check_id}


@app.get("/api/check/{check_id}/stream")
async def check_stream(check_id: str):
    if check_id not in active_checks:
        return JSONResponse({"error": "Check ID not found"}, status_code=404)

    queue = active_checks[check_id]["queue"]

    async def event_generator():
        while True:
            msg = await queue.get()
            yield msg
            if msg["event"] in ("done", "error"):
                # Clean up after stream ends
                active_checks.pop(check_id, None)
                break

    return EventSourceResponse(event_generator())

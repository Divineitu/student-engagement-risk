import io
import os
import sys
import time

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src import database, prediction, retrain
from src.preprocessing import CLASS_NAMES

app = FastAPI(title="Student Engagement Risk API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

START_TIME = time.time()

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
TRAIN_DIR = os.path.join(BASE_DIR, "data", "train")
UPLOAD_DIR = os.path.join(BASE_DIR, "data", "uploads")


@app.on_event("startup")
def startup():
    database.init_db()


@app.get("/health")
def health():
    last_retrain = database.get_last_retrain()
    return {
        "status": "ok",
        "uptime_seconds": round(time.time() - START_TIME, 1),
        "last_retrain": dict(last_retrain) if last_retrain else None,
    }


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    start = time.time()
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents))
    except Exception:
        raise HTTPException(status_code=400, detail="could not read image file")

    label, confidence, probs = prediction.predict(image)
    latency_ms = (time.time() - start) * 1000
    try:
        database.log_request("/predict", latency_ms)
    except Exception:
        pass  # a locked sqlite file under heavy load shouldn't fail the prediction itself

    return {"label": label, "confidence": confidence, "probs": probs}


@app.post("/upload")
async def upload(label: str = Form(...), files: list[UploadFile] = File(...)):
    if label not in CLASS_NAMES:
        raise HTTPException(status_code=400, detail=f"label must be one of {CLASS_NAMES}")

    class_dir = os.path.join(UPLOAD_DIR, label)
    os.makedirs(class_dir, exist_ok=True)

    saved = 0
    for f in files:
        contents = await f.read()
        dest = os.path.join(class_dir, f.filename)
        with open(dest, "wb") as out:
            out.write(contents)
        database.log_upload(f.filename, label)
        saved += 1

    return {"saved": saved, "label": label}


@app.post("/retrain")
def trigger_retrain():
    result = retrain.run_retrain()
    if result["status"] == "ok":
        prediction.reload_model()
    return result


@app.get("/stats")
def stats():
    train_counts = {}
    for class_name in CLASS_NAMES:
        class_dir = os.path.join(TRAIN_DIR, class_name)
        train_counts[class_name] = len(os.listdir(class_dir)) if os.path.isdir(class_dir) else 0

    return {
        "train_counts": train_counts,
        "upload_counts": database.get_upload_counts(),
        "recent_requests": [dict(r) for r in database.get_recent_requests(limit=100)],
    }

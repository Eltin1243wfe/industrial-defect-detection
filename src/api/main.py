"""
FastAPI service so the model can actually be called from something other than
a python script - postman, curl, a real frontend, whatever.

Run locally:
    uvicorn src.api.main:app --reload --port 8000

Then:
    curl -X POST -F "file=@some_image.png" http://localhost:8000/predict
"""

import base64
import io

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image

from src.inference.predictor import DefectPredictor
from src.utils.config import load_config, get_device

app = FastAPI(
    title="Industrial Defect Detection API",
    description="Serves anomaly detection predictions for product surface images",
    version="1.0.0",
)

# wide open for local dev / demo purposes, tighten this up before it ever
# sees a real deployment
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

cfg = load_config("configs/config.yaml")
device = get_device(cfg.project.device)
predictor = None   # lazy-loaded on startup, see below


class PredictionResponse(BaseModel):
    is_defective: bool
    score: float
    mode: str
    heatmap_overlay_b64: str | None = None


@app.on_event("startup")
def load_model():
    global predictor
    predictor = DefectPredictor(cfg, mode="patchcore", device=device)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(device), "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), return_heatmap: bool = True):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="uploaded file needs to be an image")

    try:
        raw = await file.read()
        image = Image.open(io.BytesIO(raw))
    except Exception:
        raise HTTPException(status_code=400, detail="couldn't decode that image, is it corrupted?")

    result = predictor.predict(image, return_heatmap=return_heatmap)

    response = {
        "is_defective": result["is_defective"],
        "score": result["score"],
        "mode": result["mode"],
        "heatmap_overlay_b64": None,
    }

    if "heatmap_overlay" in result:
        buf = io.BytesIO()
        result["heatmap_overlay"].save(buf, format="PNG")
        response["heatmap_overlay_b64"] = base64.b64encode(buf.getvalue()).decode("utf-8")

    return response

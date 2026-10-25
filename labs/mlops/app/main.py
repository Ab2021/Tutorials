"""FastAPI reference service with health, readiness, metrics, and prediction."""
from __future__ import annotations
import json
import logging
import os
import time
from collections import Counter
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import PlainTextResponse

from .model import RiskModel
from .schemas import PatientFeatures, Prediction

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("risk-service")

MODEL_PATH = os.getenv("MODEL_PATH", str(Path(__file__).with_name("model.json")))
try:
    MODEL = RiskModel(MODEL_PATH)
except Exception as exc:  # readiness should fail, not crash silently
    MODEL = None
    logger.error("model_load_failed error=%s", exc)

COUNTERS = Counter()
LATENCY_SUM = 0.0
LATENCY_COUNT = 0

app = FastAPI(title="Biomed Risk Service", version="0.1.0")


@app.middleware("http")
async def metrics_middleware(request: Request, call_next):
    global LATENCY_SUM, LATENCY_COUNT
    start = time.perf_counter()
    response = await call_next(request)
    elapsed = time.perf_counter() - start
    LATENCY_SUM += elapsed
    LATENCY_COUNT += 1
    COUNTERS[f"status_{response.status_code}"] += 1
    return response


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    return {"status": "ready", "model_version": MODEL.version}


@app.post("/predict", response_model=Prediction)
def predict(features: PatientFeatures):
    if MODEL is None:
        raise HTTPException(status_code=503, detail="model not loaded")
    risk = MODEL.predict(features.model_dump())
    decision = "review" if risk >= MODEL.threshold else "routine"
    COUNTERS["predictions_total"] += 1
    COUNTERS[f"decision_{decision}"] += 1
    logger.info(json.dumps({"event": "prediction", "model_version": MODEL.version, "decision": decision}))
    return Prediction(risk=round(risk, 4), threshold=MODEL.threshold, decision=decision, model_version=MODEL.version)


@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    lines = ["# HELP biomed_predictions_total Total predictions", "# TYPE biomed_predictions_total counter"]
    lines.append(f"biomed_predictions_total {COUNTERS['predictions_total']}")
    lines.append("# HELP biomed_http_requests_total HTTP requests by status")
    lines.append("# TYPE biomed_http_requests_total counter")
    for key, value in COUNTERS.items():
        if key.startswith("status_"):
            lines.append(f'biomed_http_requests_total{{status="{key[7:]}"}} {value}')
    lines.append("# HELP biomed_prediction_latency_seconds_sum Sum of prediction latency")
    lines.append("# TYPE biomed_prediction_latency_seconds_sum counter")
    lines.append(f"biomed_prediction_latency_seconds_sum {LATENCY_SUM:.6f}")
    lines.append("# HELP biomed_prediction_latency_seconds_count Count of predictions")
    lines.append("# TYPE biomed_prediction_latency_seconds_count counter")
    lines.append(f"biomed_prediction_latency_seconds_count {LATENCY_COUNT}")
    return "\n".join(lines) + "\n"

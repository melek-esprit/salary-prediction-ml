"""FastAPI service for salary prediction.

Run with:  uvicorn src.api.app:app --reload --port 8000
Docs at:   http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import json
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from configs import config
from src.models.predictor import get_predictor

app = FastAPI(title="Salary Prediction API", version="1.0.0")


class JobInput(BaseModel):
    title: str = Field(..., examples=["Senior Data Scientist"])
    skills: str = Field("", examples=["Python, SQL, Machine Learning, TensorFlow"])
    description: str = Field("", examples=["..."])
    location: str = Field("United States", examples=["United States"])
    country: str = Field("", examples=["United States"])
    company: str = Field("", examples=["Google"])
    level: str = Field("", examples=["Senior"])
    contract_type: str = Field("Full-time", examples=["CDI"])
    remote: bool = Field(False)


class PredictionOutput(BaseModel):
    predicted_salary: float
    lower: float
    upper: float
    currency: str = "USD"
    model: str


@app.get("/")
def root() -> dict:
    return {"service": "Salary Prediction API", "docs": "/docs"}


@app.get("/metrics")
def metrics() -> dict:
    path = config.REPORTS_DIR / "best_model.json"
    if not path.exists():
        raise HTTPException(404, "No trained model. Run training first.")
    return json.loads(path.read_text(encoding="utf-8"))


@app.get("/feature-importance")
def feature_importance(top: int = 25) -> dict:
    path = config.REPORTS_DIR / "feature_importance.json"
    if not path.exists():
        raise HTTPException(404, "Feature importance not generated yet.")
    data = json.loads(path.read_text(encoding="utf-8"))
    return {"top_features": data[:top]}


@app.post("/predict", response_model=PredictionOutput)
def predict(job: JobInput) -> PredictionOutput:
    try:
        predictor = get_predictor()
        result = predictor.predict_range(job.model_dump())
        return PredictionOutput(model=predictor.best_name, **result)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(500, f"Prediction failed: {exc}") from exc

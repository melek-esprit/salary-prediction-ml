"""Unified inference: load the best trained model and predict salaries.

Handles both CatBoost (.cbm) and sklearn-pipeline (.joblib) best models, and
maps raw user input (title, skills, location, ...) into the engineered feature
frame the models expect.
"""
from __future__ import annotations

import json
from typing import Optional

import joblib
import numpy as np
import pandas as pd

from configs import config
from src.features.build_features import add_engineered_features, get_feature_columns
from src.models.catboost_model import predict_usd

# Map common contract types to the dataset's schedule vocabulary.
_CONTRACT_MAP = {
    "cdi": "Full-time", "cdd": "Contractor", "permanent": "Full-time",
    "full-time": "Full-time", "part-time": "Part-time",
    "contract": "Contractor", "internship": "Internship",
    "freelance": "Contractor", "temporary": "Temp work",
}


class SalaryPredictor:
    """Loads the persisted best model and serves predictions."""

    def __init__(self) -> None:
        meta_path = config.REPORTS_DIR / "best_model.json"
        if not meta_path.exists():
            raise FileNotFoundError(
                "No trained model found. Run `python -m src.models.train` first.")
        self.meta = json.loads(meta_path.read_text(encoding="utf-8"))
        self.best_name: str = self.meta["best_model"]

        if self.best_name == "CatBoost":
            from catboost import CatBoostRegressor
            self.model = CatBoostRegressor()
            self.model.load_model(str(config.MODELS_DIR / "best_catboost.cbm"))
            self._is_catboost = True
        else:
            self.model = joblib.load(config.MODELS_DIR / "best_model.joblib")
            self._is_catboost = False

        # Optional quantile models for prediction intervals.
        lo_path = config.MODELS_DIR / "quantile_lower.joblib"
        hi_path = config.MODELS_DIR / "quantile_upper.joblib"
        self.lower_model = joblib.load(lo_path) if lo_path.exists() else None
        self.upper_model = joblib.load(hi_path) if hi_path.exists() else None

    def _build_frame(self, payload: dict) -> pd.DataFrame:
        """Convert a raw API payload into the engineered feature frame."""
        skills_raw = payload.get("skills", "") or ""
        if isinstance(skills_raw, list):
            skills_list = [str(s).strip().lower() for s in skills_raw]
        else:
            skills_list = [s.strip().lower() for s in str(skills_raw).split(",")
                           if s.strip()]
        contract = str(payload.get("contract_type", "")).lower()

        location = payload.get("location", "unknown") or "unknown"
        row = {
            "job_title": payload.get("title", ""),
            "job_title_short": payload.get("title", ""),
            "job_schedule_type": _CONTRACT_MAP.get(contract, "Full-time"),
            "job_country": payload.get("country", location),
            "job_location": location,
            "company_name": payload.get("company", "unknown") or "unknown",
            "job_type_skills": "",
            "job_work_from_home": bool(payload.get("remote", False)),
            "job_no_degree_mention": False,
            "job_health_insurance": False,
            "salary_rate": "year",
            "job_posted_date": pd.Timestamp.now(),
            "skills_list": skills_list,
            "skills_text": " ".join(skills_list),
            "n_skills": len(skills_list),
        }
        # Inject seniority hint from an explicit level field into the title.
        level = str(payload.get("level", "")).strip()
        if level and level.lower() not in row["job_title"].lower():
            row["job_title"] = f"{level} {row['job_title']}".strip()

        df = pd.DataFrame([row])
        return add_engineered_features(df)

    def predict(self, payload: dict) -> float:
        """Predict a single yearly salary (USD) — point estimate."""
        df = self._build_frame(payload)
        if self._is_catboost:
            pred = predict_usd(self.model, df)
        else:
            pred = self.model.predict(df[get_feature_columns()])
        return float(np.clip(pred[0], config.SALARY_MIN, config.SALARY_MAX))

    def predict_range(self, payload: dict) -> dict:
        """Predict a salary range: lower / point / upper (USD)."""
        df = self._build_frame(payload)
        cols = get_feature_columns()
        point = self.predict(payload)

        if self.lower_model is not None and self.upper_model is not None:
            lo = float(self.lower_model.predict(df[cols])[0])
            hi = float(self.upper_model.predict(df[cols])[0])
            lo, hi = min(lo, hi), max(lo, hi)
        else:
            # Fallback: +/- 15% band around the point estimate.
            lo, hi = point * 0.85, point * 1.15

        lo = float(np.clip(lo, config.SALARY_MIN, config.SALARY_MAX))
        hi = float(np.clip(hi, config.SALARY_MIN, config.SALARY_MAX))
        point = float(np.clip(point, lo, hi))
        return {"predicted_salary": round(point, 2),
                "lower": round(lo, 2), "upper": round(hi, 2)}


_PREDICTOR: Optional[SalaryPredictor] = None


def get_predictor() -> SalaryPredictor:
    """Return a cached predictor instance."""
    global _PREDICTOR
    if _PREDICTOR is None:
        _PREDICTOR = SalaryPredictor()
    return _PREDICTOR

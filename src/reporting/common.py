"""Shared helpers for report/presentation generation: load metrics."""
from __future__ import annotations

import json

import pandas as pd

from configs import config


def load_metrics() -> dict:
    """Load all computed metrics used in the report and slides."""
    reports = config.REPORTS_DIR
    out: dict = {}

    bm = reports / "best_model.json"
    out["best"] = json.loads(bm.read_text("utf-8")) if bm.exists() else {}

    im = reports / "interval_metrics.json"
    out["interval"] = json.loads(im.read_text("utf-8")) if im.exists() else {}

    cmp_csv = reports / "model_comparison.csv"
    out["comparison"] = (pd.read_csv(cmp_csv) if cmp_csv.exists()
                         else pd.DataFrame())

    cv_csv = reports / "cv_metrics.csv"
    out["cv"] = pd.read_csv(cv_csv) if cv_csv.exists() else pd.DataFrame()

    fi = reports / "feature_importance.json"
    out["features"] = json.loads(fi.read_text("utf-8")) if fi.exists() else []
    return out


def fig(name: str):
    """Return the path to a figure if it exists, else None."""
    p = config.FIGURES_DIR / name
    return p if p.exists() else None


def eda_fig(name: str):
    p = config.EDA_DIR / name
    return p if p.exists() else None

"""Quantile-regression models for salary prediction intervals (a range).

Trains LightGBM quantile regressors at the 10th and 90th percentiles (and a
50th-percentile median) on log(salary). Combined with the point estimate from
the best model, this lets the system output a salary *range* instead of a single
number, plus an empirical interval-coverage metric.

Run with:  python -m src.models.intervals
"""
from __future__ import annotations

import json

import joblib
import numpy as np
from lightgbm import LGBMRegressor

from configs import config
from src.features.build_features import wrap_log_target
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)
RS = config.RANDOM_STATE

LOWER_Q = 0.05
UPPER_Q = 0.95


def _load_lgbm_params() -> dict:
    """Reuse the Optuna-tuned LightGBM params if available."""
    meta = config.REPORTS_DIR / "best_model.json"
    if meta.exists():
        data = json.loads(meta.read_text("utf-8"))
        if "lightgbm_params" in data:
            return data["lightgbm_params"]
    return dict(n_estimators=700, learning_rate=0.05, num_leaves=63,
                subsample=0.8, colsample_bytree=0.8, min_child_samples=20)


def _quantile_model(alpha: float, params: dict):
    return wrap_log_target(LGBMRegressor(
        objective="quantile", alpha=alpha, **params,
        random_state=RS, n_jobs=-1, verbose=-1))


def main() -> None:
    splits = make_splits()
    params = _load_lgbm_params()

    logger.info("Training quantile models (q%.0f, q%.0f) ...",
                LOWER_Q * 100, UPPER_Q * 100)
    lower = _quantile_model(LOWER_Q, params).fit(splits.X_train, splits.y_train)
    upper = _quantile_model(UPPER_Q, params).fit(splits.X_train, splits.y_train)

    lo = lower.predict(splits.X_test)
    hi = upper.predict(splits.X_test)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)

    y = splits.y_test.to_numpy()
    coverage = float(np.mean((y >= lo) & (y <= hi)) * 100)
    mean_width = float(np.mean(hi - lo))
    nominal = (UPPER_Q - LOWER_Q) * 100

    logger.info("Interval coverage: %.1f%% (nominal %.0f%%) | mean width $%s",
                coverage, nominal, f"{mean_width:,.0f}")

    joblib.dump(lower, config.MODELS_DIR / "quantile_lower.joblib")
    joblib.dump(upper, config.MODELS_DIR / "quantile_upper.joblib")
    (config.REPORTS_DIR / "interval_metrics.json").write_text(
        json.dumps({"lower_q": LOWER_Q, "upper_q": UPPER_Q,
                    "empirical_coverage_pct": coverage,
                    "nominal_coverage_pct": nominal,
                    "mean_interval_width_usd": mean_width}, indent=2),
        encoding="utf-8")


if __name__ == "__main__":
    main()

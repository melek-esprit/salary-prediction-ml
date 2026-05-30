"""Train and compare all models, then persist the best one.

Run with:  python -m src.models.train
"""
from __future__ import annotations

import json
import time

import joblib
import pandas as pd

from configs import config
from src.evaluation.metrics import format_metrics, regression_metrics
from src.models.catboost_model import build_catboost, make_pool, predict_usd
from src.models.model_zoo import get_sklearn_models
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)


def train_all() -> pd.DataFrame:
    splits = make_splits()
    results: list[dict] = []
    fitted: dict[str, object] = {}

    # ---- sklearn model zoo -------------------------------------------------
    for name, model in get_sklearn_models().items():
        t0 = time.time()
        try:
            model.fit(splits.X_train, splits.y_train)
            pred = model.predict(splits.X_val)
            m = regression_metrics(splits.y_val, pred)
            m.update(model=name, seconds=round(time.time() - t0, 1))
            results.append(m)
            fitted[name] = model
            logger.info(format_metrics(name, m) + f" ({m['seconds']}s)")
        except Exception as exc:  # noqa: BLE001
            logger.error("%s failed: %s", name, exc)

    # ---- CatBoost (native cat + text) -------------------------------------
    t0 = time.time()
    cb = build_catboost()
    train_pool = make_pool(splits.X_train, splits.y_train)
    val_pool = make_pool(splits.X_val, splits.y_val)
    cb.fit(train_pool, eval_set=val_pool)
    pred = predict_usd(cb, splits.X_val)
    m = regression_metrics(splits.y_val, pred)
    m.update(model="CatBoost", seconds=round(time.time() - t0, 1))
    results.append(m)
    fitted["CatBoost"] = cb
    logger.info(format_metrics("CatBoost", m) + f" ({m['seconds']}s)")

    # ---- rank + pick best by validation RMSE ------------------------------
    res_df = (pd.DataFrame(results)
              .sort_values("RMSE")
              .reset_index(drop=True)
              [["model", "MAE", "RMSE", "R2", "MAPE", "seconds"]])
    res_df.to_csv(config.REPORTS_DIR / "model_comparison.csv", index=False)
    logger.info("\n%s", res_df.to_string(index=False))

    best_name = res_df.iloc[0]["model"]
    best_model = fitted[best_name]
    logger.info("BEST on validation: %s", best_name)

    # ---- evaluate best on the held-out test set ---------------------------
    if best_name == "CatBoost":
        test_pred = predict_usd(best_model, splits.X_test)
        best_model.save_model(str(config.MODELS_DIR / "best_catboost.cbm"))
    else:
        test_pred = best_model.predict(splits.X_test)
    joblib.dump(best_model, config.MODELS_DIR / "best_model.joblib")

    test_m = regression_metrics(splits.y_test, test_pred)
    logger.info("TEST  | " + format_metrics(best_name, test_m))

    (config.REPORTS_DIR / "best_model.json").write_text(
        json.dumps({"best_model": best_name,
                    "validation": res_df.iloc[0].to_dict(),
                    "test": test_m}, indent=2, default=float),
        encoding="utf-8")
    return res_df


if __name__ == "__main__":
    train_all()

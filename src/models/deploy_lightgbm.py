"""Build the compact, deployable LightGBM model committed to the repository.

The full stacking ensemble (`build_stack`) reaches R2~0.587 but weighs ~806 MB
(it carries a 400-tree ExtraTrees plus three preprocessing copies) and exceeds
GitHub's 100 MB file limit. A single LightGBM, trained with the same
Optuna-tuned hyper-parameters, scores within ~0.005 R2 of the ensemble while
fitting in a few MB — so it ships as the out-of-the-box model.

Run with:  python -m src.models.deploy_lightgbm
"""
from __future__ import annotations

import json

import joblib
from lightgbm import LGBMRegressor

from configs import config
from src.evaluation.metrics import format_metrics, regression_metrics
from src.features.build_features import wrap_log_target
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)
RS = config.RANDOM_STATE


def main() -> None:
    meta_path = config.REPORTS_DIR / "best_model.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    lgbm_params = meta["lightgbm_params"]
    ensemble_val = meta.get("validation")
    ensemble_test = meta.get("test")

    splits = make_splits()
    logger.info("Training deployable LightGBM with tuned params ...")
    model = wrap_log_target(
        LGBMRegressor(**lgbm_params, random_state=RS, n_jobs=-1, verbose=-1))
    model.fit(splits.X_train, splits.y_train)

    val_m = regression_metrics(splits.y_val, model.predict(splits.X_val))
    test_m = regression_metrics(splits.y_test, model.predict(splits.X_test))
    logger.info("VAL  | " + format_metrics("LightGBM (deployable)", val_m))
    logger.info("TEST | " + format_metrics("LightGBM (deployable)", test_m))

    out = config.MODELS_DIR / "best_model.joblib"
    joblib.dump(model, out, compress=3)
    size_mb = out.stat().st_size / 1024 / 1024
    logger.info("Saved deployable model -> %s (%.2f MB)", out, size_mb)

    # CatBoost is not the served model; remove any stale pointer file.
    cb = config.MODELS_DIR / "best_catboost.cbm"
    if cb.exists():
        cb.unlink()

    meta_path.write_text(json.dumps({
        "best_model": "LightGBM",
        "note": (
            "Deployable single LightGBM committed to the repo (a few MB). "
            "The full stacking ensemble scores marginally higher but is ~806 MB "
            "and is regenerated via `python -m src.models.advanced`."),
        "lightgbm_params": lgbm_params,
        "xgboost_params": meta.get("xgboost_params"),
        "validation": {**val_m, "model": "LightGBM"},
        "test": test_m,
        "stacking_ensemble_reference": {
            "validation": ensemble_val, "test": ensemble_test},
    }, indent=2, default=float), encoding="utf-8")
    logger.info("Updated best_model.json -> deployed model = LightGBM")


if __name__ == "__main__":
    main()

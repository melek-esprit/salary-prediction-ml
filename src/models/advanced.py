"""Advanced modeling: Optuna-tuned LightGBM + stacking ensemble.

Tunes LightGBM, builds a StackingRegressor (ExtraTrees + tuned LightGBM +
XGBoost, Ridge meta-learner), and if it beats the current best model on the
validation set it is promoted to the project's best model.

Run with:  python -m src.models.advanced
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import optuna
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from configs import config
from src.evaluation.metrics import format_metrics, regression_metrics
from src.features.build_features import wrap_log_target
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)
RS = config.RANDOM_STATE


def tune_lightgbm(splits, n_trials: int = 20) -> dict:
    """Optuna search minimizing validation RMSE (USD)."""

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 900),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            num_leaves=trial.suggest_int("num_leaves", 31, 220),
            max_depth=trial.suggest_int("max_depth", 4, 12),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        )
        model = wrap_log_target(
            LGBMRegressor(**params, random_state=RS, n_jobs=-1, verbose=-1))
        model.fit(splits.X_train, splits.y_train)
        pred = model.predict(splits.X_val)
        return regression_metrics(splits.y_val, pred)["RMSE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best LightGBM RMSE (val): $%s", f"{study.best_value:,.0f}")
    logger.info("Best params: %s", study.best_params)
    return study.best_params


def tune_xgboost(splits, n_trials: int = 25) -> dict:
    """Optuna search for XGBoost minimizing validation RMSE (USD)."""

    def objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators=trial.suggest_int("n_estimators", 300, 1000),
            learning_rate=trial.suggest_float("learning_rate", 0.01, 0.12, log=True),
            max_depth=trial.suggest_int("max_depth", 3, 11),
            subsample=trial.suggest_float("subsample", 0.6, 1.0),
            colsample_bytree=trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            min_child_weight=trial.suggest_int("min_child_weight", 1, 20),
            gamma=trial.suggest_float("gamma", 1e-3, 5.0, log=True),
        )
        model = wrap_log_target(XGBRegressor(
            **params, random_state=RS, n_jobs=-1, tree_method="hist"))
        model.fit(splits.X_train, splits.y_train)
        pred = model.predict(splits.X_val)
        return regression_metrics(splits.y_val, pred)["RMSE"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    logger.info("Best XGBoost RMSE (val): $%s", f"{study.best_value:,.0f}")
    return study.best_params


def build_stack(lgbm_params: dict, xgb_params: dict) -> StackingRegressor:
    base = [
        ("extratrees", wrap_log_target(ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=RS))),
        ("lightgbm", wrap_log_target(LGBMRegressor(
            **lgbm_params, random_state=RS, n_jobs=-1, verbose=-1))),
        ("xgboost", wrap_log_target(XGBRegressor(
            **xgb_params, random_state=RS, n_jobs=-1, tree_method="hist"))),
    ]
    return StackingRegressor(estimators=base, final_estimator=Ridge(alpha=1.0),
                             cv=5, n_jobs=1)


def main(lgbm_trials: int = 40, xgb_trials: int = 25) -> None:
    splits = make_splits()

    logger.info("Tuning LightGBM with Optuna (%d trials) ...", lgbm_trials)
    lgbm_params = tune_lightgbm(splits, n_trials=lgbm_trials)

    logger.info("Tuning XGBoost with Optuna (%d trials) ...", xgb_trials)
    xgb_params = tune_xgboost(splits, n_trials=xgb_trials)

    logger.info("Fitting stacking ensemble (5-fold internal CV) ...")
    stack = build_stack(lgbm_params, xgb_params)
    stack.fit(splits.X_train, splits.y_train)

    val_m = regression_metrics(splits.y_val, stack.predict(splits.X_val))
    test_m = regression_metrics(splits.y_test, stack.predict(splits.X_test))
    logger.info("VAL  | " + format_metrics("StackingEnsemble", val_m))
    logger.info("TEST | " + format_metrics("StackingEnsemble", test_m))

    # Promote if it beats the current best validation RMSE.
    meta_path = config.REPORTS_DIR / "best_model.json"
    current = json.loads(meta_path.read_text("utf-8")) if meta_path.exists() else None
    current_rmse = current["validation"]["RMSE"] if current else float("inf")

    if val_m["RMSE"] < current_rmse:
        logger.info("Stacking improves over previous best -> promoting.")
        joblib.dump(stack, config.MODELS_DIR / "best_model.joblib")
        cb = config.MODELS_DIR / "best_catboost.cbm"
        if cb.exists():
            cb.unlink()
        meta_path.write_text(json.dumps({
            "best_model": "StackingEnsemble",
            "lightgbm_params": lgbm_params,
            "xgboost_params": xgb_params,
            "validation": {**val_m, "model": "StackingEnsemble"},
            "test": test_m,
        }, indent=2, default=float), encoding="utf-8")
    else:
        logger.info("Stacking did not beat the current best; keeping previous.")


if __name__ == "__main__":
    main()

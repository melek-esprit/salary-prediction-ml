"""5-fold cross-validated evaluation of the tuned models (robust metrics).

Reports mean +/- std of R2, MAE, RMSE, MAPE across folds on the full salaried
dataset. This is a more trustworthy headline than a single train/test split and
is the recommended number to cite in an academic report.

Run with:  python -m src.evaluation.cross_validate
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import KFold, cross_validate
from xgboost import XGBRegressor

from configs import config
from src.features.build_features import get_feature_columns, wrap_log_target
from src.preprocessing.prepare import load_prepared
from src.utils.logger import get_logger

logger = get_logger(__name__)
RS = config.RANDOM_STATE


def _mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def _load_params() -> tuple[dict, dict]:
    meta = config.REPORTS_DIR / "best_model.json"
    data = json.loads(meta.read_text("utf-8")) if meta.exists() else {}
    lgbm = data.get("lightgbm_params", dict(
        n_estimators=700, learning_rate=0.05, num_leaves=63))
    xgb = data.get("xgboost_params", dict(
        n_estimators=600, learning_rate=0.05, max_depth=6))
    return lgbm, xgb


def main() -> None:
    df = load_prepared()
    X = df[get_feature_columns()]
    y = df[config.TARGET_YEAR]
    lgbm_params, xgb_params = _load_params()

    models = {
        "LightGBM (tuned)": wrap_log_target(LGBMRegressor(
            **lgbm_params, random_state=RS, n_jobs=-1, verbose=-1)),
        "XGBoost (tuned)": wrap_log_target(XGBRegressor(
            **xgb_params, random_state=RS, n_jobs=-1, tree_method="hist")),
        "ExtraTrees": wrap_log_target(ExtraTreesRegressor(
            n_estimators=400, max_features="sqrt", n_jobs=-1, random_state=RS)),
    }

    cv = KFold(n_splits=config.N_SPLITS_CV, shuffle=True, random_state=RS)
    scoring = {"r2": "r2",
               "mae": "neg_mean_absolute_error",
               "rmse": "neg_root_mean_squared_error"}

    rows = []
    for name, model in models.items():
        res = cross_validate(model, X, y, cv=cv, scoring=scoring, n_jobs=1)
        row = {
            "model": name,
            "R2_mean": float(np.mean(res["test_r2"])),
            "R2_std": float(np.std(res["test_r2"])),
            "MAE_mean": float(-np.mean(res["test_mae"])),
            "RMSE_mean": float(-np.mean(res["test_rmse"])),
        }
        rows.append(row)
        logger.info("%s | R2 %.4f +/- %.4f | MAE $%s | RMSE $%s",
                    name, row["R2_mean"], row["R2_std"],
                    f"{row['MAE_mean']:,.0f}", f"{row['RMSE_mean']:,.0f}")

    out = pd.DataFrame(rows).sort_values("R2_mean", ascending=False)
    out.to_csv(config.REPORTS_DIR / "cv_metrics.csv", index=False)
    (config.REPORTS_DIR / "cv_metrics.json").write_text(
        out.to_json(orient="records", indent=2), encoding="utf-8")
    logger.info("Saved %d-fold CV metrics.", config.N_SPLITS_CV)


if __name__ == "__main__":
    main()

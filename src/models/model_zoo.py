"""Model zoo: sklearn-pipeline regressors with log-target wrapping.

Each model is a full pipeline (preprocessor + estimator) wrapped in a
TransformedTargetRegressor so it trains on log1p(salary) and predicts in USD.
CatBoost is built separately (it consumes raw categorical + text features).
"""
from __future__ import annotations

import numpy as np
from lightgbm import LGBMRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import (ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestRegressor)
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from configs import config
from src.features.build_features import build_preprocessor

RS = config.RANDOM_STATE


def _wrap(estimator) -> TransformedTargetRegressor:
    """Wrap (preprocessor + estimator) to train on log1p(salary)."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)


def get_sklearn_models() -> dict[str, TransformedTargetRegressor]:
    """Return the dictionary of sklearn-based models to benchmark."""
    return {
        "LinearRegression": _wrap(LinearRegression()),
        "Ridge": _wrap(Ridge(alpha=2.0, random_state=RS)),
        "Lasso": _wrap(Lasso(alpha=0.001, random_state=RS, max_iter=5000)),
        "ElasticNet": _wrap(ElasticNet(alpha=0.001, l1_ratio=0.5,
                                       random_state=RS, max_iter=5000)),
        "RandomForest": _wrap(RandomForestRegressor(
            n_estimators=200, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", n_jobs=-1, random_state=RS)),
        "ExtraTrees": _wrap(ExtraTreesRegressor(
            n_estimators=200, max_features="sqrt", n_jobs=-1, random_state=RS)),
        # Bounded GBM: native impls (XGB/LGBM) dominate, so keep this light.
        "GradientBoosting": _wrap(GradientBoostingRegressor(
            n_estimators=150, max_depth=3, subsample=0.7,
            max_features=0.1, random_state=RS)),
        "XGBoost": _wrap(XGBRegressor(
            n_estimators=600, learning_rate=0.05, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RS, tree_method="hist")),
        "LightGBM": _wrap(LGBMRegressor(
            n_estimators=800, learning_rate=0.05, num_leaves=63,
            subsample=0.8, colsample_bytree=0.8, n_jobs=-1,
            random_state=RS, verbose=-1)),
    }

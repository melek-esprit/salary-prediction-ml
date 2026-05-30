"""CatBoost regressor using native categorical + text features on log target."""
from __future__ import annotations

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool

from configs import config
from src.features.build_features import (NUMERIC_FEATURES, OHE_FEATURES,
                                         TARGET_ENC_FEATURES, SKILLS_TEXT,
                                         TITLE_TEXT)

CAT_FEATURES = OHE_FEATURES + TARGET_ENC_FEATURES
TEXT_FEATURES = [SKILLS_TEXT, TITLE_TEXT]
CB_COLUMNS = NUMERIC_FEATURES + CAT_FEATURES + TEXT_FEATURES


def make_pool(X: pd.DataFrame, y: pd.Series | None = None) -> Pool:
    """Build a CatBoost Pool with categorical + text features declared."""
    X = X[CB_COLUMNS].copy()
    for c in CAT_FEATURES:
        X[c] = X[c].astype(str)
    for c in TEXT_FEATURES:
        X[c] = X[c].fillna("").astype(str)
    label = np.log1p(y) if y is not None else None
    return Pool(X, label=label, cat_features=CAT_FEATURES,
               text_features=TEXT_FEATURES)


def build_catboost(**overrides) -> CatBoostRegressor:
    params = dict(
        iterations=600,
        learning_rate=0.06,
        depth=6,
        loss_function="RMSE",
        eval_metric="RMSE",
        random_seed=config.RANDOM_STATE,
        l2_leaf_reg=3.0,
        early_stopping_rounds=60,
        verbose=200,
    )
    params.update(overrides)
    return CatBoostRegressor(**params)


def predict_usd(model: CatBoostRegressor, X: pd.DataFrame) -> np.ndarray:
    """Predict and invert the log transform back to USD."""
    pool = make_pool(X)
    return np.expm1(model.predict(pool))

"""Prepare modeling data: clean target, engineer features, split train/val/test."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from configs import config
from src.data.loader import get_salaried_dataset
from src.features.build_features import add_engineered_features, get_feature_columns
from src.utils.logger import get_logger

logger = get_logger(__name__)
TARGET = config.TARGET_YEAR


@dataclass
class DataSplits:
    X_train: pd.DataFrame
    X_val: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_val: pd.Series
    y_test: pd.Series

    def describe(self) -> str:
        return (f"train={len(self.X_train):,} | val={len(self.X_val):,} | "
                f"test={len(self.X_test):,}")


def load_prepared() -> pd.DataFrame:
    """Load salaried data, cap outliers, engineer features."""
    df = get_salaried_dataset("year")
    before = len(df)
    df = df[(df[TARGET] >= config.SALARY_MIN) &
            (df[TARGET] <= config.SALARY_MAX)].copy()
    logger.info("Outlier cap [%s, %s]: kept %s of %s rows",
                config.SALARY_MIN, config.SALARY_MAX, f"{len(df):,}", f"{before:,}")
    df = add_engineered_features(df)
    return df


def make_splits(df: pd.DataFrame | None = None) -> DataSplits:
    """Create train/val/test splits with the engineered feature columns."""
    if df is None:
        df = load_prepared()

    cols = get_feature_columns()
    X = df[cols].copy()
    y = df[TARGET].copy()

    X_tr, X_test, y_tr, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.RANDOM_STATE)
    val_ratio = config.VAL_SIZE / (1 - config.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_tr, y_tr, test_size=val_ratio, random_state=config.RANDOM_STATE)

    splits = DataSplits(X_train, X_val, X_test, y_train, y_val, y_test)
    logger.info("Splits: %s", splits.describe())
    return splits


if __name__ == "__main__":
    s = make_splits()
    print(s.describe())
    print(s.X_train.dtypes)

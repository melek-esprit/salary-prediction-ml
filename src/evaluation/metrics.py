"""Regression metrics computed on the original (USD) salary scale."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Return MAE, RMSE, R2 and MAPE."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)
    return {"MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}


def format_metrics(name: str, m: dict[str, float]) -> str:
    return (f"{name:28s} | MAE ${m['MAE']:>9,.0f} | RMSE ${m['RMSE']:>9,.0f} "
            f"| R2 {m['R2']:.4f} | MAPE {m['MAPE']:.1f}%")

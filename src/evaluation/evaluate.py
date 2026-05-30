"""Evaluate the best model on the test set and generate diagnostic plots.

Run with:  python -m src.evaluation.evaluate
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from configs import config
from src.evaluation.metrics import format_metrics, regression_metrics
from src.models.catboost_model import predict_usd
from src.models.model_zoo import get_sklearn_models  # noqa: F401 (ensures imports)
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _load_best():
    meta = json.loads((config.REPORTS_DIR / "best_model.json").read_text("utf-8"))
    name = meta["best_model"]
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(config.MODELS_DIR / "best_catboost.cbm"))
        return name, model, True
    return name, joblib.load(config.MODELS_DIR / "best_model.joblib"), False


def main() -> None:
    name, model, is_cb = _load_best()
    splits = make_splits()
    y_test = splits.y_test.to_numpy()
    pred = (predict_usd(model, splits.X_test) if is_cb
            else model.predict(splits.X_test))

    m = regression_metrics(y_test, pred)
    logger.info("TEST | " + format_metrics(name, m))
    resid = y_test - pred

    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    axes[0].scatter(y_test, pred, s=8, alpha=0.3, color="#2b6cb0")
    lim = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    axes[0].plot(lim, lim, "r--", lw=1)
    axes[0].set(xlabel="Actual (USD)", ylabel="Predicted (USD)",
                title=f"Predicted vs Actual — {name}")

    axes[1].scatter(pred, resid, s=8, alpha=0.3, color="#c05621")
    axes[1].axhline(0, color="r", ls="--", lw=1)
    axes[1].set(xlabel="Predicted (USD)", ylabel="Residual",
                title="Residuals")

    sns.histplot(resid, bins=60, kde=True, ax=axes[2], color="#2f855a")
    axes[2].set(title="Error distribution", xlabel="Residual (USD)")

    fig.suptitle(f"{name}  |  MAE ${m['MAE']:,.0f}  RMSE ${m['RMSE']:,.0f}  "
                 f"R2 {m['R2']:.3f}  MAPE {m['MAPE']:.1f}%", fontsize=13)
    fig.tight_layout()
    out = config.FIGURES_DIR / "evaluation_diagnostics.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", out)


if __name__ == "__main__":
    main()

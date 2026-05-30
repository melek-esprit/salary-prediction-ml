"""Model explainability: feature importance + SHAP (CatBoost native).

Run with:  python -m src.evaluation.explain
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs import config
from src.models.catboost_model import CB_COLUMNS, make_pool
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")


def _save_importance(names, values, title, fname) -> list[dict]:
    order = np.argsort(values)[::-1]
    names = np.asarray(names)[order]
    values = np.asarray(values)[order]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.barplot(x=values[:25], y=names[:25], ax=ax, color="#2b6cb0")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / fname, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return [{"feature": str(n), "importance": float(v)}
            for n, v in zip(names, values)]


def explain_catboost(model, splits) -> list[dict]:
    pool = make_pool(splits.X_test, splits.y_test)
    imp = model.get_feature_importance(pool)
    ranked = _save_importance(CB_COLUMNS, imp,
                              "CatBoost feature importance",
                              "feature_importance.png")

    # SHAP values (CatBoost native) on a sample for a summary plot.
    sample = splits.X_test.sample(min(1000, len(splits.X_test)),
                                  random_state=config.RANDOM_STATE)
    shap = model.get_feature_importance(make_pool(sample), type="ShapValues")
    mean_abs = np.abs(shap[:, :-1]).mean(axis=0)
    _save_importance(CB_COLUMNS, mean_abs,
                     "Mean |SHAP| per feature (CatBoost)",
                     "shap_summary.png")
    return ranked


def explain_sklearn(model, splits) -> list[dict]:
    """Model-agnostic permutation importance over the raw input columns.

    Works for any sklearn estimator (single pipeline or stacking ensemble).
    """
    from sklearn.inspection import permutation_importance

    from src.features.build_features import get_feature_columns

    cols = get_feature_columns()
    X = splits.X_test[cols]
    result = permutation_importance(
        model, X, splits.y_test, n_repeats=5,
        random_state=config.RANDOM_STATE, scoring="r2", n_jobs=1)
    return _save_importance(cols, result.importances_mean,
                            "Permutation importance (drop in R2)",
                            "feature_importance.png")


def main() -> None:
    meta = json.loads((config.REPORTS_DIR / "best_model.json").read_text("utf-8"))
    name = meta["best_model"]
    splits = make_splits()

    if name == "CatBoost":
        from catboost import CatBoostRegressor
        model = CatBoostRegressor()
        model.load_model(str(config.MODELS_DIR / "best_catboost.cbm"))
        ranked = explain_catboost(model, splits)
    else:
        model = joblib.load(config.MODELS_DIR / "best_model.joblib")
        ranked = explain_sklearn(model, splits)

    (config.REPORTS_DIR / "feature_importance.json").write_text(
        json.dumps(ranked, indent=2), encoding="utf-8")
    logger.info("Top features: %s",
                ", ".join(r["feature"] for r in ranked[:10]))


if __name__ == "__main__":
    main()

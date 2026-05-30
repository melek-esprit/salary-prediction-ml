"""Visualize prediction intervals and their empirical coverage on the test set.

Produces:
  - interval_coverage.png : sorted actual salaries with the predicted
    5th-95th percentile band, highlighting points inside vs outside the band.

Run with:  python -m src.evaluation.interval_plot
"""
from __future__ import annotations

import json

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from configs import config
from src.preprocessing.prepare import make_splits
from src.utils.logger import get_logger

logger = get_logger(__name__)


def main() -> None:
    lo_path = config.MODELS_DIR / "quantile_lower.joblib"
    hi_path = config.MODELS_DIR / "quantile_upper.joblib"
    if not (lo_path.exists() and hi_path.exists()):
        logger.error("Quantile models not found. Run src.models.intervals first.")
        return

    lower = joblib.load(lo_path)
    upper = joblib.load(hi_path)
    splits = make_splits()

    lo = lower.predict(splits.X_test)
    hi = upper.predict(splits.X_test)
    lo, hi = np.minimum(lo, hi), np.maximum(lo, hi)
    y = splits.y_test.to_numpy()

    inside = (y >= lo) & (y <= hi)
    coverage = float(np.mean(inside) * 100)
    width = float(np.mean(hi - lo))

    order = np.argsort(y)
    y_s, lo_s, hi_s, in_s = y[order], lo[order], hi[order], inside[order]
    x = np.arange(len(y_s))

    fig, ax = plt.subplots(figsize=(13, 6))
    ax.fill_between(x, lo_s, hi_s, color="#90cdf4", alpha=0.5,
                    label="Predicted 5th-95th pct interval")
    ax.scatter(x[in_s], y_s[in_s], s=6, color="#2f855a", label="Actual (inside)")
    ax.scatter(x[~in_s], y_s[~in_s], s=10, color="#c53030",
               label="Actual (outside)")
    ax.set_xlabel("Test samples (sorted by actual salary)")
    ax.set_ylabel("Yearly salary (USD)")
    ax.set_title(f"Prediction intervals — coverage {coverage:.1f}% "
                 f"(nominal 90%), mean width ${width:,.0f}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    out = config.FIGURES_DIR / "interval_coverage.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("Coverage %.1f%% | saved %s", coverage, out.name)


if __name__ == "__main__":
    main()

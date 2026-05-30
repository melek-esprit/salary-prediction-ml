"""End-to-end orchestrator for the salary prediction project.

Usage:
    python main.py eda        # run exploratory data analysis
    python main.py train      # train & compare all models, save the best
    python main.py evaluate   # test-set diagnostics for the best model
    python main.py explain    # feature importance + SHAP
    python main.py all        # eda -> train -> evaluate -> explain
    python main.py db-check    # verify the SQL Server connection
"""
from __future__ import annotations

import sys

from src.utils.logger import get_logger

logger = get_logger("main")

STEPS = {
    "eda": ("src.eda.run_eda", "main"),
    "train": ("src.models.train", "train_all"),
    "evaluate": ("src.evaluation.evaluate", "main"),
    "explain": ("src.evaluation.explain", "main"),
}


def _run(step: str) -> None:
    module_name, func_name = STEPS[step]
    module = __import__(module_name, fromlist=[func_name])
    logger.info("=== running step: %s ===", step)
    getattr(module, func_name)()


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "all"

    if cmd == "db-check":
        from src.db.connection import test_connection
        test_connection()
        return

    if cmd == "all":
        for step in ("eda", "train", "evaluate", "explain"):
            _run(step)
    elif cmd in STEPS:
        _run(cmd)
    else:
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main(sys.argv)

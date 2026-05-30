"""Data loading and basic parsing for data_jobs.csv (clean source of truth)."""
from __future__ import annotations

import ast
from typing import Optional

import pandas as pd

from configs import config
from src.utils.logger import get_logger

logger = get_logger(__name__)


def parse_skill_list(value: object) -> list[str]:
    """Parse the stringified python list in `job_skills` into a clean list."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    if isinstance(value, list):
        return [str(s).strip().lower() for s in value]
    try:
        parsed = ast.literal_eval(str(value))
        if isinstance(parsed, (list, tuple)):
            return [str(s).strip().lower() for s in parsed]
    except (ValueError, SyntaxError):
        pass
    return []


def load_raw(usecols: Optional[list[str]] = None) -> pd.DataFrame:
    """Load the raw CSV."""
    logger.info("Loading raw CSV from %s", config.RAW_CSV)
    df = pd.read_csv(config.RAW_CSV, usecols=usecols)
    logger.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


def get_salaried_dataset(rate: str = "year") -> pd.DataFrame:
    """Return rows with a usable salary target, with skills parsed.

    Args:
        rate: "year" -> uses salary_year_avg, "hour" -> uses salary_hour_avg.
    """
    target = config.TARGET_YEAR if rate == "year" else config.TARGET_HOUR
    df = load_raw()

    before = len(df)
    df = df[df[target].notna() & (df[target] > 0)].copy()
    logger.info("Filtered to %s salaried rows (from %s) using '%s'",
                f"{len(df):,}", f"{before:,}", target)

    # Parse skills into list + a clean space-joined string for TF-IDF.
    df["skills_list"] = df[config.SKILLS_COL].apply(parse_skill_list)
    df["skills_text"] = df["skills_list"].apply(lambda xs: " ".join(xs))
    df["n_skills"] = df["skills_list"].apply(len)

    # Parse posting date.
    df[config.DATE_COL] = pd.to_datetime(df[config.DATE_COL], errors="coerce")

    df = df.reset_index(drop=True)
    return df


if __name__ == "__main__":
    d = get_salaried_dataset("year")
    print(d[[config.TARGET_YEAR, "n_skills", "skills_text"]].head())
    print("shape:", d.shape)

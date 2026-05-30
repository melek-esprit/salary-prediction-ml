"""Feature engineering and the sklearn preprocessing pipeline.

Produces:
  - an engineered modeling DataFrame (numeric + categorical + text columns)
  - a ColumnTransformer that vectorizes everything into a numeric matrix
    (used by linear / RandomForest / XGBoost / LightGBM models).

CatBoost is handled separately (it consumes raw categorical + text natively).
"""
from __future__ import annotations

import ast
import re

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler, TargetEncoder

from configs import config

# Keyword-based seniority scoring from the free-text title.
_SENIORITY = {
    r"\b(intern|internship|junior|jr|entry|graduate|trainee)\b": -1,
    r"\b(senior|sr|lead|principal|staff|head|manager|director|chief|vp)\b": 2,
    r"\b(mid|intermediate)\b": 0,
}

# Structured skill categories parsed from `job_type_skills`.
SKILL_CATEGORIES = ["programming", "databases", "cloud", "libraries",
                    "analyst_tools", "webframeworks", "other"]
_CAT_FEATURES_NUM = [f"cat_{c}" for c in SKILL_CATEGORIES] + ["n_skill_categories"]

NUMERIC_FEATURES = [
    "n_skills", "title_len", "title_word_count", "remote_flag",
    "no_degree_flag", "health_flag", "seniority_score", "posted_month",
] + _CAT_FEATURES_NUM
OHE_FEATURES = ["job_title_short", "job_schedule_type", "salary_rate"]
# High-cardinality columns handled with leakage-safe target encoding.
TARGET_ENC_FEATURES = ["job_country", "company_name", "job_location"]
SKILLS_TEXT = "skills_text"
TITLE_TEXT = "job_title"


def _parse_skill_categories(value: object) -> dict[str, int]:
    """Parse the `job_type_skills` dict-string into per-category skill counts."""
    counts = {f"cat_{c}": 0 for c in SKILL_CATEGORIES}
    counts["n_skill_categories"] = 0
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return counts
    try:
        parsed = ast.literal_eval(str(value))
    except (ValueError, SyntaxError):
        return counts
    if not isinstance(parsed, dict):
        return counts
    n_cats = 0
    for cat, items in parsed.items():
        n = len(items) if isinstance(items, (list, tuple)) else 0
        key = f"cat_{cat}"
        if key in counts:
            counts[key] = n
        else:
            counts["cat_other"] += n
        if n > 0:
            n_cats += 1
    counts["n_skill_categories"] = n_cats
    return counts


def _to_int_flag(series: pd.Series) -> pd.Series:
    """Coerce a boolean-ish column to 0/1."""
    return (series.astype(str).str.lower()
            .isin({"true", "1", "yes", "t"}).astype(int))


def _seniority_score(title: str) -> int:
    title = str(title).lower()
    for pattern, score in _SENIORITY.items():
        if re.search(pattern, title):
            return score
    return 0


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived numeric/categorical features. Operates on a copy."""
    df = df.copy()

    df["title_len"] = df[TITLE_TEXT].fillna("").str.len()
    df["title_word_count"] = df[TITLE_TEXT].fillna("").str.split().apply(len)
    df["remote_flag"] = _to_int_flag(df["job_work_from_home"])
    df["no_degree_flag"] = _to_int_flag(df["job_no_degree_mention"])
    df["health_flag"] = _to_int_flag(df["job_health_insurance"])
    df["seniority_score"] = df[TITLE_TEXT].apply(_seniority_score)

    if "job_posted_date" in df.columns:
        dt = pd.to_datetime(df["job_posted_date"], errors="coerce")
        df["posted_month"] = dt.dt.month.fillna(0).astype(int)
    else:
        df["posted_month"] = 0

    # Structured skill-category counts from job_type_skills.
    if "job_type_skills" in df.columns:
        cat_df = pd.DataFrame(
            df["job_type_skills"].apply(_parse_skill_categories).tolist(),
            index=df.index)
    else:
        cat_df = pd.DataFrame(0, index=df.index, columns=_CAT_FEATURES_NUM)
    for col in _CAT_FEATURES_NUM:
        df[col] = cat_df.get(col, 0).astype(int)

    # Fill categorical/text holes so transformers don't choke.
    for col in OHE_FEATURES + TARGET_ENC_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str)
    df[TITLE_TEXT] = df[TITLE_TEXT].fillna("")
    if SKILLS_TEXT not in df.columns:
        df[SKILLS_TEXT] = ""
    df[SKILLS_TEXT] = df[SKILLS_TEXT].fillna("")
    return df


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer: numeric scaling + OHE + target-encoding + TF-IDF."""
    numeric = Pipeline([("scale", StandardScaler())])

    ohe = OneHotEncoder(handle_unknown="ignore", min_frequency=10,
                        sparse_output=True)

    target_enc = TargetEncoder(target_type="continuous", random_state=config.RANDOM_STATE)

    skills_tfidf = TfidfVectorizer(
        max_features=config.TFIDF_MAX_FEATURES,
        ngram_range=(1, 1),         # skills are already tokens
        token_pattern=r"[^\s]+",    # keep tokens like "power bi" parts / c++
    )
    title_tfidf = TfidfVectorizer(
        max_features=1500,
        ngram_range=config.TFIDF_NGRAM_RANGE,
        stop_words="english",
        min_df=5,
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric, NUMERIC_FEATURES),
            ("ohe", ohe, OHE_FEATURES),
            ("tgt", target_enc, TARGET_ENC_FEATURES),
            ("skills", skills_tfidf, SKILLS_TEXT),
            ("title", title_tfidf, TITLE_TEXT),
        ],
        remainder="drop",
        sparse_threshold=0.3,
    )


def get_feature_columns() -> list[str]:
    """All input columns required by the preprocessor."""
    return (NUMERIC_FEATURES + OHE_FEATURES + TARGET_ENC_FEATURES
            + [SKILLS_TEXT, TITLE_TEXT])


def wrap_log_target(estimator) -> TransformedTargetRegressor:
    """Wrap (preprocessor + estimator) to train on log1p(salary), predict USD."""
    pipe = Pipeline([("prep", build_preprocessor()), ("model", estimator)])
    return TransformedTargetRegressor(regressor=pipe, func=np.log1p,
                                      inverse_func=np.expm1)

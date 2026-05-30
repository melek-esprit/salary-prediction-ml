"""Central configuration for the salary prediction project.

Single source of truth for paths, column names, and modeling settings.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_CSV = PROJECT_ROOT / "data_jobs.csv"           # clean source of truth
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
EDA_DIR = OUTPUTS_DIR / "eda"
MODELS_DIR = OUTPUTS_DIR / "models"
REPORTS_DIR = OUTPUTS_DIR / "reports"
FIGURES_DIR = OUTPUTS_DIR / "figures"

for _d in (DATA_DIR, OUTPUTS_DIR, EDA_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# Columns (as they appear in data_jobs.csv)
# --------------------------------------------------------------------------- #
TARGET_YEAR = "salary_year_avg"
TARGET_HOUR = "salary_hour_avg"

TEXT_COLS = ["job_title", "job_skills"]
CATEGORICAL_COLS = [
    "job_title_short",
    "job_schedule_type",
    "job_country",
    "job_work_from_home",
    "job_no_degree_mention",
    "job_health_insurance",
    "salary_rate",
]
SKILLS_COL = "job_skills"
TITLE_COL = "job_title"
DATE_COL = "job_posted_date"

# --------------------------------------------------------------------------- #
# Modeling
# --------------------------------------------------------------------------- #
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15          # of the remaining train after test split
N_SPLITS_CV = 5
LOG_TARGET = True        # train on log1p(salary), invert for metrics

# Outlier capping on the yearly salary target (USD)
SALARY_MIN = 15_000
SALARY_MAX = 400_000     # cap extreme outliers (99th pct ~274k)

# TF-IDF
TFIDF_MAX_FEATURES = 3000
TFIDF_NGRAM_RANGE = (1, 2)

# Top-N skills to one-hot as explicit features
TOP_N_SKILLS = 80

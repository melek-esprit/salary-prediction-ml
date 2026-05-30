"""Automated EDA for the salaried job-postings dataset.

Generates figures (PNG) into outputs/eda and a markdown insights report.
Run with:  python -m src.eda.run_eda
"""
from __future__ import annotations

from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from configs import config
from src.data.loader import get_salaried_dataset
from src.utils.logger import get_logger

logger = get_logger(__name__)
sns.set_theme(style="whitegrid")
TARGET = config.TARGET_YEAR


def _save(fig: plt.Figure, name: str) -> str:
    path = config.EDA_DIR / name
    fig.tight_layout()
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    logger.info("saved %s", path.name)
    return name


def salary_distribution(df: pd.DataFrame, lines: list[str]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    sns.histplot(df[TARGET], bins=60, kde=True, ax=axes[0], color="#2b6cb0")
    axes[0].set_title("Yearly salary (USD)")
    sns.histplot(np.log1p(df[TARGET]), bins=60, kde=True, ax=axes[1], color="#2f855a")
    axes[1].set_title("log1p(salary) — used as model target")
    _save(fig, "01_salary_distribution.png")

    skew_raw = df[TARGET].skew()
    skew_log = np.log1p(df[TARGET]).skew()
    lines.append(f"- Salary is right-skewed (skew={skew_raw:.2f}). "
                 f"After log1p, skew drops to {skew_log:.2f}, justifying a "
                 f"**log-transformed target**.")


def missing_and_overview(df_raw: pd.DataFrame, lines: list[str]) -> None:
    miss = (df_raw.isna().mean() * 100).sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(9, 6))
    miss[miss > 0].plot.barh(ax=ax, color="#c05621")
    ax.set_title("Missing values (%) — full dataset")
    ax.set_xlabel("% missing")
    _save(fig, "02_missing_values.png")
    lines.append(f"- Full dataset: **{len(df_raw):,} rows**. "
                 f"Salary present on only ~{df_raw[TARGET].notna().mean()*100:.1f}% "
                 f"of rows -> modeling uses the salaried subset.")


def salary_by_category(df: pd.DataFrame, col: str, fname: str,
                       lines: list[str], top: int = 12) -> None:
    order = (df.groupby(col)[TARGET].median()
             .sort_values(ascending=False).head(top).index)
    sub = df[df[col].isin(order)]
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.boxplot(data=sub, x=col, y=TARGET, order=order, ax=ax, showfliers=False)
    ax.set_title(f"Salary by {col}")
    ax.tick_params(axis="x", rotation=40)
    for lbl in ax.get_xticklabels():
        lbl.set_ha("right")
    _save(fig, fname)


def remote_analysis(df: pd.DataFrame, lines: list[str]) -> None:
    grp = df.groupby("job_work_from_home")[TARGET].median()
    if len(grp) == 2:
        diff = grp.get(True, np.nan) - grp.get(False, np.nan)
        lines.append(f"- Remote jobs median salary differs by "
                     f"**${diff:,.0f}** vs non-remote.")


def top_skills(df: pd.DataFrame, lines: list[str]) -> None:
    counter: Counter = Counter()
    salary_by_skill: dict[str, list] = {}
    for skills, sal in zip(df["skills_list"], df[TARGET]):
        for s in set(skills):
            counter[s] += 1
            salary_by_skill.setdefault(s, []).append(sal)

    common = counter.most_common(25)
    names = [c[0] for c in common]
    counts = [c[1] for c in common]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=counts, y=names, ax=ax, color="#2b6cb0")
    ax.set_title("Top 25 most frequent skills (salaried jobs)")
    _save(fig, "06_top_skills.png")

    # Highest-paying skills (min 100 occurrences).
    paying = {s: np.median(v) for s, v in salary_by_skill.items() if len(v) >= 100}
    top_pay = sorted(paying.items(), key=lambda x: x[1], reverse=True)[:20]
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.barplot(x=[p[1] for p in top_pay], y=[p[0] for p in top_pay],
                ax=ax, color="#2f855a")
    ax.set_title("Top 20 highest-paying skills (median, >=100 postings)")
    ax.set_xlabel("median salary (USD)")
    _save(fig, "07_highest_paying_skills.png")
    lines.append("- Highest-paying skills (median): "
                 + ", ".join(f"{s} (${v:,.0f})" for s, v in top_pay[:8]))


def correlation_numeric(df: pd.DataFrame, lines: list[str]) -> None:
    df = df.copy()
    df["remote"] = df["job_work_from_home"].astype("category").cat.codes
    num = df[[TARGET, "n_skills", "remote"]].corr()
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(num, annot=True, cmap="coolwarm", center=0, ax=ax)
    ax.set_title("Numeric correlations")
    _save(fig, "08_correlation.png")


def main() -> None:
    df = get_salaried_dataset("year")
    df_raw = pd.read_csv(config.RAW_CSV, usecols=[TARGET, "job_skills",
                         "job_title_short", "job_country", "job_schedule_type",
                         "job_work_from_home", "salary_rate"])

    df = df[(df[TARGET] >= config.SALARY_MIN)].copy()

    lines: list[str] = ["# EDA Insights — Salary Prediction\n"]
    lines.append(f"Usable salaried rows (yearly): **{len(df):,}**\n")

    desc = df[TARGET].describe(percentiles=[.05, .25, .5, .75, .95])
    lines.append("## Salary summary (USD)\n")
    lines.append("```\n" + desc.round(0).to_string() + "\n```")

    missing_and_overview(df_raw, lines)
    salary_distribution(df, lines)
    salary_by_category(df, "job_title_short", "03_salary_by_title.png", lines)
    salary_by_category(df, "job_country", "04_salary_by_country.png", lines)
    salary_by_category(df, "job_schedule_type", "05_salary_by_schedule.png", lines)
    remote_analysis(df, lines)
    top_skills(df, lines)
    correlation_numeric(df, lines)

    lines.append(f"\n- Average skills per salaried posting: "
                 f"**{df['n_skills'].mean():.1f}**.")
    lines.append(f"- Skills present on **{(df['n_skills']>0).mean()*100:.1f}%** "
                 f"of salaried rows.")

    report = config.REPORTS_DIR / "eda_insights.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote insights report -> %s", report)
    logger.info("EDA complete: 8 figures in %s", config.EDA_DIR)


if __name__ == "__main__":
    main()

"""Quick diagnostics on the salaried subset and skill coverage."""
from __future__ import annotations

import pandas as pd
from sqlalchemy import text

from src.db.connection import get_engine

QUERIES = {
    "salaried_year_count": "SELECT COUNT(*) AS n FROM Fact_Jobs WHERE salary_average_year IS NOT NULL;",
    "salaried_hour_count": "SELECT COUNT(*) AS n FROM Fact_Jobs WHERE salary_average_hour IS NOT NULL;",
    "fact_total": "SELECT COUNT(*) AS n FROM Fact_Jobs;",
    "facts_with_any_skill": """
        SELECT COUNT(DISTINCT b.job_fact_id) AS n
        FROM Bridge_Job_Skills b;
    """,
    "salaried_facts_with_skill": """
        SELECT COUNT(DISTINCT f.job_fact_id) AS n
        FROM Fact_Jobs f
        JOIN Bridge_Job_Skills b ON f.job_fact_id = b.job_fact_id
        WHERE f.salary_average_year IS NOT NULL;
    """,
    "avg_skills_per_salaried_job": """
        SELECT AVG(CAST(cnt AS FLOAT)) AS avg_skills FROM (
            SELECT f.job_fact_id, COUNT(b.skill_id) AS cnt
            FROM Fact_Jobs f
            JOIN Bridge_Job_Skills b ON f.job_fact_id = b.job_fact_id
            WHERE f.salary_average_year IS NOT NULL
            GROUP BY f.job_fact_id
        ) t;
    """,
}


def main() -> None:
    with get_engine().connect() as conn:
        for name, q in QUERIES.items():
            val = conn.execute(text(q)).scalar()
            print(f"{name:32s}: {val}")

        print("\nTop 15 job titles (salaried):")
        df = pd.read_sql(text("""
            SELECT j.job_title_short, COUNT(*) AS n,
                   AVG(f.salary_average_year) AS avg_salary
            FROM Fact_Jobs f JOIN Dim_Job j ON f.job_id = j.job_id
            WHERE f.salary_average_year IS NOT NULL
            GROUP BY j.job_title_short ORDER BY n DESC;
        """), conn)
        print(df.head(15).to_string(index=False))


if __name__ == "__main__":
    main()

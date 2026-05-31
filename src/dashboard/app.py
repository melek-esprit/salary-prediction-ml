"""Streamlit dashboard for the salary prediction system.

Run with:  streamlit run src/dashboard/app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure the project root is importable when launched via `streamlit run`
# (Streamlit only puts the script's own folder on sys.path).
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd
import streamlit as st

from configs import config

st.set_page_config(page_title="Salary Prediction", layout="wide", page_icon="$")


@st.cache_resource
def _load_predictor():
    from src.models.predictor import get_predictor
    return get_predictor()


def _read_json(path):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


st.title("Job Salary Prediction System")
st.caption("ML + NLP salary estimation from job-posting metadata and skills.")

tab_predict, tab_models, tab_features, tab_eda = st.tabs(
    ["Predict", "Model comparison", "Feature importance", "Data insights"])

with tab_predict:
    st.subheader("Estimate a yearly salary (USD)")
    c1, c2 = st.columns(2)
    with c1:
        title = st.text_input("Job title", "Senior Data Scientist")
        skills = st.text_input("Skills (comma-separated)",
                               "python, sql, machine learning, tensorflow, aws")
        location = st.text_input("Location / Country", "United States")
        company = st.text_input("Company (optional)", "")
    with c2:
        level = st.selectbox("Seniority", ["", "Junior", "Mid", "Senior",
                                           "Lead", "Principal"], index=3)
        contract = st.selectbox("Contract type",
                                ["Full-time", "Contractor", "Part-time",
                                 "Internship"], index=0)
        remote = st.checkbox("Remote", value=False)

    if st.button("Predict salary", type="primary"):
        try:
            predictor = _load_predictor()
            payload = {"title": title, "skills": skills, "location": location,
                       "company": company, "level": level,
                       "contract_type": contract, "remote": remote}
            r = predictor.predict_range(payload)
            st.metric("Estimated yearly salary",
                      f"${r['predicted_salary']:,.0f}")
            st.success(f"Likely range: **${r['lower']:,.0f}  –  "
                       f"${r['upper']:,.0f}**")
            c_lo, c_mid, c_hi = st.columns(3)
            c_lo.metric("Lower (10th pct)", f"${r['lower']:,.0f}")
            c_mid.metric("Point estimate", f"${r['predicted_salary']:,.0f}")
            c_hi.metric("Upper (90th pct)", f"${r['upper']:,.0f}")
            st.caption(f"Point model: {predictor.best_name} | "
                       f"range via quantile regression")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not predict: {exc}")
            st.info("Train first: `python main.py train` then "
                    "`python -m src.models.intervals`")

with tab_models:
    st.subheader("Model comparison (validation set)")
    csv = config.REPORTS_DIR / "model_comparison.csv"
    if csv.exists():
        df = pd.read_csv(csv)
        st.dataframe(df, use_container_width=True)
        st.bar_chart(df.set_index("model")["R2"])
    else:
        st.info("Run training to populate this table.")

    cv_csv = config.REPORTS_DIR / "cv_metrics.csv"
    if cv_csv.exists():
        st.subheader("5-fold cross-validated metrics (robust)")
        st.dataframe(pd.read_csv(cv_csv), use_container_width=True)

    interval_img = config.FIGURES_DIR / "interval_coverage.png"
    if interval_img.exists():
        st.subheader("Prediction-interval coverage")
        st.image(str(interval_img), use_container_width=True)

with tab_features:
    st.subheader("What drives salary?")
    fi = config.FIGURES_DIR / "feature_importance.png"
    shap = config.FIGURES_DIR / "shap_summary.png"
    if fi.exists():
        st.image(str(fi), use_container_width=True)
    if shap.exists():
        st.image(str(shap), use_container_width=True)
    if not fi.exists():
        st.info("Run `python -m src.evaluation.explain`.")

with tab_eda:
    st.subheader("Exploratory data analysis")
    report = config.REPORTS_DIR / "eda_insights.md"
    if report.exists():
        st.markdown(report.read_text(encoding="utf-8"))
    for img in sorted(config.EDA_DIR.glob("*.png")):
        st.image(str(img), caption=img.stem, use_container_width=True)

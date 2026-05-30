# Job Salary Prediction System (ML + NLP)

End-to-end machine-learning system that predicts **yearly job salaries (USD)**
from job-posting metadata and skills, sourced from a Microsoft SQL Server data
warehouse (`DatawarehouseDB`) and the underlying `data_jobs.csv` dataset.

The project connects to SQL Server, runs automated EDA, engineers NLP + tabular
features, trains and compares many models, tunes the best with Optuna, builds a
stacking ensemble, explains predictions, and serves them through a FastAPI API
and a Streamlit dashboard.

---

## 1. Dataset & target

- Source: `data_jobs.csv` (~785k data/analytics job postings, 2023).
- **Target:** `salary_year_avg` (USD). Only ~22k rows have a real salary, so the
  model trains on that **salaried subset**.
- The salary is right-skewed (skew 1.75), so models train on **`log1p(salary)`**
  and predictions are inverted back to USD.

> **Note on the warehouse:** profiling `DatawarehouseDB` revealed ETL issues in
> the original load (missing salaries stored as `0`, an incomplete skills bridge,
> and a broken `Fact_Jobs -> Dim_Job` link). The clean CSV is therefore used as
> the training source of truth, while the live SQL Server connection is kept for
> warehouse access and BI. See `src/db/` for the connection + schema tools.

## 2. Project structure

```
.
├── configs/            # central configuration
├── data/               # (local data)
├── outputs/
│   ├── eda/            # EDA figures
│   ├── figures/        # evaluation + importance plots
│   ├── models/         # saved best model
│   └── reports/        # comparison table, metrics, insights
├── src/
│   ├── db/             # SQL Server connection, schema inspection, diagnostics
│   ├── data/           # CSV loader + skills parsing
│   ├── preprocessing/  # target cleaning + train/val/test splits
│   ├── features/       # feature engineering + sklearn preprocessor
│   ├── models/         # model zoo, CatBoost, training, Optuna+stacking, predictor
│   ├── evaluation/     # metrics, diagnostics plots, explainability (SHAP)
│   ├── api/            # FastAPI service
│   └── dashboard/      # Streamlit app
├── main.py             # orchestrator
├── requirements.txt
└── .env                # DB credentials (not committed)
```

## 3. Setup

```bash
python -m pip install -r requirements.txt
```

Configure the database in `.env` (already set for Windows Authentication):

```
DB_SERVER=LAPTOP-N87ER6LI\MSSQLSERVER01
DB_NAME=DatawarehouseDB
DB_DRIVER=ODBC Driver 17 for SQL Server
DB_TRUSTED_CONNECTION=yes
```

## 4. How to run

```bash
# Verify the SQL Server connection
python main.py db-check

# Inspect the warehouse schema
python -m src.db.inspect_schema

# Full pipeline: EDA -> train -> evaluate -> explain
python main.py all

# Or step by step
python main.py eda
python main.py train        # trains the model zoo, picks best by validation RMSE
python -m src.models.advanced   # Optuna tuning + stacking ensemble (promotes if better)
python -m src.models.intervals  # quantile models for the salary range
python main.py evaluate
python main.py explain
```

Serve the API and dashboard:

```bash
uvicorn src.api.app:app --reload --port 8000     # docs at /docs
streamlit run src/dashboard/app.py
```

## 4b. Results

Best model: **stacking ensemble** (ExtraTrees + Optuna-tuned LightGBM + XGBoost,
Ridge meta-learner), trained on `log1p(salary)`.

| Metric (held-out test) | Value |
|------------------------|-------|
| R²                     | 0.59  |
| MAE                    | ~$20,950 |
| RMSE                   | ~$29,612 |
| MAPE                   | 18.7% |

The system outputs a **salary range** via quantile regression (5th–95th
percentile), with ~85% empirical coverage on the test set. Top salary drivers:
job title, company, seniority, skills, country, and location.

## 5. Modeling approach

1. **Features**
   - Numeric/engineered: `n_skills`, title length/word-count, remote flag,
     no-degree flag, health-insurance flag, seniority score, posting month, and
     structured skill-category counts (programming, cloud, databases, ...).
   - Categorical: `job_title_short`, `job_schedule_type`, `salary_rate`
     (one-hot) and high-cardinality `job_country`, `company_name`, `job_location`
     (leakage-safe target encoding).
   - Text (NLP): TF-IDF over **skills** and over the **job title** (1–2 grams).
   - CatBoost additionally consumes the raw categorical + text columns natively.
2. **Models compared:** Linear / Ridge / Lasso / ElasticNet, RandomForest,
   ExtraTrees, GradientBoosting, XGBoost, LightGBM, CatBoost.
3. **Tuning & ensembling:** Optuna tunes LightGBM; a stacking ensemble
   (ExtraTrees + tuned LightGBM + XGBoost with a Ridge meta-learner) is built and
   promoted if it beats the best single model.
4. **Evaluation:** MAE, RMSE, R², MAPE on a held-out test set, plus
   predicted-vs-actual, residual, and error-distribution plots.
5. **Explainability:** CatBoost native feature importance + SHAP, and
   model-agnostic permutation importance for the ensemble.

## 6. API example

`POST /predict`

```json
{
  "title": "Senior Data Scientist",
  "skills": "Python, SQL, Machine Learning, TensorFlow",
  "location": "United States",
  "level": "Senior",
  "contract_type": "Full-time",
  "remote": true
}
```

Response (point estimate + range):

```json
{
  "predicted_salary": 157494.0,
  "lower": 92335.0,
  "upper": 202264.0,
  "currency": "USD",
  "model": "StackingEnsemble"
}
```

## 7. Outputs

- `outputs/reports/model_comparison.csv` — all model metrics
- `outputs/reports/best_model.json` — chosen model + test metrics
- `outputs/reports/eda_insights.md` — EDA findings
- `outputs/eda/*.png`, `outputs/figures/*.png` — visualizations
- `outputs/models/best_model.joblib` — the deployed model

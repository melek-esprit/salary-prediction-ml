"""Smoke test: load the best model and predict on a few example jobs."""
from src.models.predictor import get_predictor

p = get_predictor()
print("Loaded best model:", p.best_name)

tests = [
    {"title": "Senior Data Scientist",
     "skills": "python, sql, machine learning, tensorflow, aws",
     "location": "United States", "level": "Senior",
     "contract_type": "Full-time", "remote": True},
    {"title": "Data Analyst", "skills": "excel, sql, tableau",
     "location": "India", "level": "Junior",
     "contract_type": "Full-time", "remote": False},
    {"title": "Machine Learning Engineer",
     "skills": "python, pytorch, kafka, spark, aws",
     "location": "United States", "level": "Lead",
     "contract_type": "Full-time", "remote": True},
]
for t in tests:
    r = p.predict_range(t)
    print(f"  {t['title']:28s} ({t['level']:7s}) -> "
          f"${r['predicted_salary']:,.0f}  "
          f"[${r['lower']:,.0f} - ${r['upper']:,.0f}]")

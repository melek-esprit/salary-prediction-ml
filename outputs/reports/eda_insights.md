# EDA Insights — Salary Prediction

Usable salaried rows (yearly): **22,003**

## Salary summary (USD)

```
count     22003.0
mean     123286.0
std       48312.0
min       15000.0
5%        57500.0
25%       90000.0
50%      115000.0
75%      150000.0
95%      203000.0
max      960000.0
```
- Full dataset: **785,741 rows**. Salary present on only ~2.8% of rows → modeling uses the salaried subset.
- Salary is right-skewed (skew=1.75). After log1p, skew drops to -0.18, justifying a **log-transformed target**.
- Remote jobs median salary differs by **$13,830** vs non-remote.
- Highest-paying skills (median): mongo ($173,500), cassandra ($150,000), golang ($147,500), scala ($147,500), redis ($147,500), kafka ($147,500), pytorch ($147,500), neo4j ($147,500)

- Average skills per salaried posting: **5.4**.
- Skills present on **91.7%** of salaried rows.
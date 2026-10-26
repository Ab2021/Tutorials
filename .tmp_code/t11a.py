CRITERIA = {
    "fit": 0.30,
    "reproducibility": 0.20,
    "licence": 0.20,
    "team_skill": 0.15,
    "cost": 0.10,
    "operations": 0.05,
}
STACKS = {
    "DuckDB + Python + Parquet": {"fit": 4, "reproducibility": 4, "licence": 5, "team_skill": 5, "cost": 5, "operations": 4},
    "PostgreSQL + dbt + Airflow": {"fit": 5, "reproducibility": 5, "licence": 5, "team_skill": 3, "cost": 3, "operations": 3},
    "Spark + Delta Lake": {"fit": 5, "reproducibility": 4, "licence": 5, "team_skill": 2, "cost": 2, "operations": 2},
}

print(f"{'Stack':<32}{'Weighted score':>16}")
for name, scores in STACKS.items():
    total = sum(CRITERIA[k] * scores[k] for k in CRITERIA)
    print(f"{name:<32}{total:>16.2f}")
print()
print("Weights sum:", sum(CRITERIA.values()))
print("Scores are 1 to 5, higher is better.")

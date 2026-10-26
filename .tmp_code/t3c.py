def required_admissions(prevalence, predictors=25, epv=10, inflation=1.0):
    events = predictors * epv
    n = events / prevalence
    return round(n * inflation)

print(f"{'Stratum':<30}{'Prevalence':>12}{'Required n':>12}")
rows = [
    ("All admissions", 0.22),
    ("Age 65 and older", 0.24),
    ("Age under 65", 0.17),
    ("Medicare", 0.25),
    ("Commercial", 0.14),
    ("Rural site", 0.12),
]
for name, prev in rows:
    print(f"{name:<30}{prev:>12.0%}{required_admissions(prev):>12,}")
print()
print("Assumptions: 25 predictors, 10 events per predictor, no design inflation.")
print("Add 20 to 40 percent for subgroup analyses and missing data.")

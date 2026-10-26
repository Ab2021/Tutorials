FEATURES = [
    ("age", -87600.0, "registration"),
    ("triage_vitals", 0.1, "triage"),
    ("initial_lactate", 1.2, "lab"),
    ("wbc", 1.4, "lab"),
    ("culture_collected", 2.0, "micro"),
    ("antibiotic_first_dose", 2.3, "eMAR"),
    ("urine_output_6h", 6.0, "flowsheet"),
    ("creatinine_repeat", 7.5, "lab"),
    ("lactate_repeat", 8.2, "lab"),
    ("vasopressor_start", 9.4, "eMAR"),
]
PREDICTION_HOUR = 6.0
late = [f for f in FEATURES if f[1] > PREDICTION_HOUR]
print(f"Prediction time: hour {PREDICTION_HOUR:.0f}")
print(f"Features audited: {len(FEATURES)}")
print(f"Available at or before prediction time: {len(FEATURES) - len(late)}")
print(f"Late features that would leak: {len(late)}")
for name, t, src in late:
    print(f"  LEAK {name:<24} first at {t:5.1f} h ({src})")
print()
print("Boundary case: urine_output_6h is timestamped exactly at the prediction")
print("time. Include only if the value is complete before the alert fires.")

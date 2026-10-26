COUNTS = {
    "patients (all)": 364627,
    "hospitalizations": 546028,
    "ICU stays": 94458,
    "unique ICU patients": 65366,
    "ED-only patients": 141175,
}

print("MIMIC-IV v3.1 unit-of-analysis audit")
print(f"{'Entity':<24}{'Count':>10}")
for name, n in COUNTS.items():
    print(f"{name:<24}{n:>10,}")

hospitalized = COUNTS["patients (all)"] - COUNTS["ED-only patients"]
print()
print(f"Hospitalized patients (all minus ED-only): {hospitalized:,}")
print(f"Hospitalizations per hospitalized patient: {COUNTS['hospitalizations']/hospitalized:.2f}")
print(f"ICU stays per hospitalization: {COUNTS['ICU stays']/COUNTS['hospitalizations']:.3f}")
print(f"ICU stays per ICU patient: {COUNTS['ICU stays']/COUNTS['unique ICU patients']:.2f}")
print()
print("Interpretation:")
print("  A row in the hosp module is a hospitalization, not a patient.")
print("  Counting rows as patients overstates the cohort by the repeat-admission factor.")
print("  Only about one in six hospitalizations has an ICU stay.")

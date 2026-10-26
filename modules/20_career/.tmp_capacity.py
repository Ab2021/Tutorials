"""Capacity and latency budget calculator, standard library only.

Run: python capacity_budget.py
Computes request rates, storage, allowed downtime, and a latency budget for a
clinical scoring system. All inputs are stated in one place so they can be
changed and re-run during interview preparation.
"""

BEDS = 1200
SCORES_PER_BED_PER_DAY = 6
PEAK_FACTOR = 8
PAYLOAD_KB = 2
FEATURES = 150
BYTES_PER_FEATURE = 8
UPTIME_TARGET = 0.999
LATENCY_BUDGET_SECONDS = 900  # 15 minutes

# Stage name, seconds allocated.
LATENCY_STAGES = [
    ("device or lab ingestion", 120),
    ("validation and deduplication", 30),
    ("feature computation", 300),
    ("model inference", 60),
    ("delivery to worklist", 60),
]


def main():
    scores_per_day = BEDS * SCORES_PER_BED_PER_DAY
    avg_qps = scores_per_day / 86_400
    peak_qps = avg_qps * PEAK_FACTOR
    daily_payload_mb = scores_per_day * PAYLOAD_KB / 1024
    patient_hours_per_year = BEDS * 24 * 365
    feature_gb_per_year = patient_hours_per_year * FEATURES * BYTES_PER_FEATURE / 1e9
    downtime_minutes = (1 - UPTIME_TARGET) * 525_600

    print("WORKLOAD")
    print(f"  beds: {BEDS}")
    print(f"  scores per day: {scores_per_day:,}")
    print(f"  average requests per second: {avg_qps:.3f}")
    print(f"  peak requests per second: {peak_qps:.2f}")
    print(f"  score payload per day: {daily_payload_mb:.1f} MB")
    print(f"  patient-hours per year: {patient_hours_per_year:,.0f}")
    print(f"  feature storage per year: {feature_gb_per_year:.1f} GB (before overhead)")
    print()
    print("AVAILABILITY")
    print(f"  target: {UPTIME_TARGET:.3%}")
    print(f"  allowed downtime: {downtime_minutes:.1f} minutes per year "
          f"({downtime_minutes / 60:.1f} hours)")

    print()
    print("LATENCY BUDGET")
    print(f"  budget: {LATENCY_BUDGET_SECONDS} seconds")
    total = 0
    for name, seconds in LATENCY_STAGES:
        total += seconds
        print(f"  {name:30s} {seconds:4d} s  ({seconds / LATENCY_BUDGET_SECONDS:5.1%})")
    print(f"  {'total':30s} {total:4d} s")
    slack = LATENCY_BUDGET_SECONDS - total
    print(f"  slack: {slack} s ({slack / LATENCY_BUDGET_SECONDS:.1%})")
    if slack < 0:
        print("  WARNING: the design exceeds its latency budget.")
    elif slack < 0.1 * LATENCY_BUDGET_SECONDS:
        print("  NOTE: less than 10 percent slack; any stage regression breaks the budget.")

    print()
    print("WHAT THE NUMBERS IMPLY")
    print("  A sub-1 QPS peak does not justify a streaming platform.")
    print("  A 15-minute budget is met by a 5-minute micro-batch with recomputation on late labs.")
    print("  The binding constraint is interface freshness, not compute.")


if __name__ == "__main__":
    main()

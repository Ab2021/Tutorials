"""Threshold and capacity analysis, standard library only.

Run: python threshold_capacity.py
Given a score distribution and an outcome count per score bin, computes the
operating characteristics of every cumulative threshold from the top down and
selects the lowest threshold that fits a stated capacity.
"""

# (score bin lower bound, patients per month, readmissions per month)
BINS = [
    (0.0, 310, 10),
    (0.1, 180, 12),
    (0.2, 120, 13),
    (0.3, 80, 12),
    (0.4, 50, 11),
    (0.5, 20, 8),
    (0.6, 20, 9),
    (0.7, 12, 7),
    (0.8, 6, 5),
    (0.9, 2, 2),
]

CAPACITY_PER_MONTH = 60


def cumulative_from_top(bins):
    """Yield thresholds from the highest bin downward with cumulative stats."""
    total_patients = sum(n for _, n, _ in bins)
    total_events = sum(e for _, _, e in bins)
    patients = 0
    events = 0
    rows = []
    for lower, n, e in reversed(bins):
        patients += n
        events += e
        rows.append(
            {
                "threshold": lower,
                "alerts": patients,
                "tp": events,
                "ppv": events / patients,
                "sensitivity": events / total_events,
                "flag_rate": patients / total_patients,
            }
        )
    return rows, total_patients, total_events


def choose(rows, capacity):
    """Lowest threshold whose alert volume does not exceed capacity."""
    eligible = [r for r in rows if r["alerts"] <= capacity]
    if not eligible:
        return None
    return min(eligible, key=lambda r: r["threshold"])


def main():
    rows, total_patients, total_events = cumulative_from_top(BINS)
    prevalence = total_events / total_patients
    print(f"eligible patients per month: {total_patients}")
    print(f"readmissions per month:      {total_events}")
    print(f"prevalence:                  {prevalence:.1%}")
    print(f"capacity:                    {CAPACITY_PER_MONTH}")
    print(f"maximum flag rate:           {CAPACITY_PER_MONTH / total_patients:.1%}")
    print()
    print(f"  {'threshold':>9s} {'alerts':>7s} {'TP':>4s} {'PPV':>7s} {'sens':>7s} {'flag rate':>10s} {'fits':>6s}")
    for r in sorted(rows, key=lambda x: x["threshold"], reverse=True):
        fits = "yes" if r["alerts"] <= CAPACITY_PER_MONTH else "no"
        print(f"  {r['threshold']:9.1f} {r['alerts']:7d} {r['tp']:4d} "
              f"{r['ppv']:7.1%} {r['sensitivity']:7.1%} {r['flag_rate']:10.1%} {fits:>6s}")

    chosen = choose(rows, CAPACITY_PER_MONTH)
    print()
    print("CHOSEN OPERATING POINT")
    print(f"  threshold: {chosen['threshold']:.2f}")
    print(f"  alerts per month: {chosen['alerts']}")
    print(f"  PPV: {chosen['ppv']:.1%}")
    print(f"  sensitivity: {chosen['sensitivity']:.1%}")
    print(f"  readmissions reached: {chosen['tp']} of {total_events}")

    relative_effect = 0.20
    absolute = chosen["ppv"] * relative_effect
    nnt = 1 / absolute if absolute else float("inf")
    prevented = chosen["alerts"] * absolute * 12
    print()
    print("INTERVENTION ARITHMETIC (assuming a 20% relative reduction)")
    print(f"  absolute reduction among enrolled: {absolute:.1%}")
    print(f"  number needed to treat: {nnt:.1f}")
    print(f"  readmissions prevented per year: {prevented:.1f}")

    print()
    print("CAPACITY SENSITIVITY")
    for capacity in (20, 60, 100, 200):
        pick = choose(rows, capacity)
        if pick is None:
            print(f"  capacity {capacity:3d}: no threshold fits")
            continue
        print(f"  capacity {capacity:3d}: threshold {pick['threshold']:.1f}, "
              f"PPV {pick['ppv']:.1%}, sensitivity {pick['sensitivity']:.1%}")


if __name__ == "__main__":
    main()

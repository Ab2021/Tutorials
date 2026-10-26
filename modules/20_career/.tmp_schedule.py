"""Interview practice scheduler, standard library only.

Run: python interview_schedule.py
Allocates a fixed number of preparation hours across interview stages by
expected loss: decision weight x failure probability x cost of failure.
A floor gives every stage at least one hour. No cap is applied by the model;
Section 7 explains why the candidate applies caps manually.
"""

STAGES = [
    # name, decision weight, failure probability, cost of failure (1-5)
    ("Recruiter screen", 0.10, 0.10, 2),
    ("Hiring manager", 0.30, 0.70, 5),
    ("Technical screen", 0.25, 0.30, 3),
    ("Clinical case", 0.20, 0.50, 4),
    ("Panel presentation", 0.15, 0.80, 5),
]

TOTAL_HOURS = 40.0
FLOOR = 1.0


def priority(stage):
    _, weight, fail, cost = stage
    return weight * fail * cost


def allocate(total, stages, floor=FLOOR):
    scores = {s[0]: priority(s) for s in stages}
    names = [s[0] for s in stages]
    weight_sum = sum(scores.values())
    hours = {n: max(floor, total * scores[n] / weight_sum) for n in names}
    # Rescale the stages above the floor so the total is exact.
    above = [n for n in names if hours[n] > floor]
    fixed_total = floor * (len(names) - len(above))
    scale = (total - fixed_total) / sum(hours[n] for n in above)
    for n in above:
        hours[n] *= scale
    return hours


def main():
    print("PRIORITY SCORES")
    print(f"  {'stage':20s} {'weight':>6s} {'p_fail':>6s} {'cost':>5s} {'priority':>8s}")
    for name, weight, fail, cost in STAGES:
        print(f"  {name:20s} {weight:6.2f} {fail:6.2f} {cost:5d} {priority((name, weight, fail, cost)):8.3f}")
    print(f"  total priority: {sum(priority(s) for s in STAGES):.3f}")

    hours = allocate(TOTAL_HOURS, STAGES)
    print()
    print(f"ALLOCATION OF {TOTAL_HOURS:.0f} HOURS (proportional, floor 1.0)")
    print(f"  {'stage':20s} {'hours':>6s} {'share':>6s}")
    for name, _, _, _ in STAGES:
        print(f"  {name:20s} {hours[name]:6.1f} {hours[name] / TOTAL_HOURS:6.1%}")
    print(f"  {'total':20s} {sum(hours.values()):6.1f}")

    revised = [
        ("Recruiter screen", 0.10, 0.10, 2),
        ("Hiring manager", 0.30, 0.40, 5),
        ("Technical screen", 0.25, 0.30, 3),
        ("Clinical case", 0.20, 0.50, 4),
        ("Panel presentation", 0.15, 0.80, 5),
    ]
    hours2 = allocate(TOTAL_HOURS, revised)
    print()
    print("EFFECT OF LOWERING HIRING MANAGER FAILURE PROBABILITY 0.70 -> 0.40")
    for name, _, _, _ in STAGES:
        print(f"  {name:20s} {hours[name]:5.1f} -> {hours2[name]:5.1f} ({hours2[name] - hours[name]:+.1f})")


if __name__ == "__main__":
    main()

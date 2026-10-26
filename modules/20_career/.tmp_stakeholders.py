"""Stakeholder engagement planner for the first 90 days, standard library only.

Run: python stakeholder_plan.py
Given stakeholders scored on influence and interest, assigns an engagement
cadence, flags overdue contacts, and summarises coverage by group.
"""

STAKEHOLDERS = [
    # role, group, influence 1-5, interest 1-5, days since last contact
    ("Director of analytics", "analytics", 5, 5, 3),
    ("CMIO", "clinical leadership", 5, 4, 18),
    ("Care management director", "operations", 4, 5, 9),
    ("Quality director", "operations", 4, 3, 22),
    ("Data governance chair", "governance", 5, 3, 12),
    ("Privacy officer", "governance", 4, 2, 30),
    ("EHR reporting lead", "analytics", 3, 5, 4),
    ("Two analysts on the team", "analytics", 2, 5, 2),
    ("Nurse informatics lead", "clinical leadership", 3, 4, 16),
    ("Finance partner", "operations", 3, 2, 45),
    ("Research director", "clinical leadership", 3, 3, 27),
    ("IT platform owner", "analytics", 4, 3, 11),
    ("Population health lead", "operations", 4, 4, 20),
    ("Executive sponsor", "leadership", 5, 5, 25),
]


def cadence_days(influence, interest):
    if influence >= 4 and interest >= 4:
        return 7
    if influence >= 4:
        return 30
    if interest >= 4:
        return 30
    return 90


def main():
    rows = []
    for role, group, influence, interest, last in STAKEHOLDERS:
        cadence = cadence_days(influence, interest)
        overdue = max(0, last - cadence)
        rows.append((role, group, influence, interest, last, cadence, overdue))

    print("STAKEHOLDER ENGAGEMENT PLAN")
    print(f"  {'role':28s} {'group':20s} {'inf':>4s} {'int':>4s} {'last':>5s} {'cad':>5s} {'over':>5s}")
    for role, group, influence, interest, last, cadence, overdue in rows:
        print(f"  {role:28s} {group:20s} {influence:4d} {interest:4d} {last:5d} {cadence:5d} {overdue:5d}")

    overdue_rows = sorted([r for r in rows if r[6] > 0], key=lambda r: -r[6])
    print()
    print("OVERDUE, MOST OVERDUE FIRST")
    if not overdue_rows:
        print("  none")
    for role, group, influence, interest, last, cadence, overdue in overdue_rows:
        print(f"  {role:28s} {overdue:3d} days past a {cadence}-day cadence")

    print()
    print("COVERAGE BY GROUP")
    groups = {}
    for role, group, influence, interest, last, cadence, overdue in rows:
        g = groups.setdefault(group, {"n": 0, "overdue": 0, "high": 0})
        g["n"] += 1
        g["overdue"] += 1 if overdue > 0 else 0
        g["high"] += 1 if influence >= 4 else 0
    for group, g in sorted(groups.items()):
        print(f"  {group:22s} people {g['n']:2d}  high influence {g['high']:2d}  overdue {g['overdue']:2d}")

    print()
    print("WEEKLY TIME BUDGET")
    weekly = sum(1 for r in rows if r[5] == 7)
    monthly = sum(1 for r in rows if r[5] == 30)
    quarterly = sum(1 for r in rows if r[5] == 90)
    hours = weekly * 0.5 + monthly * 0.5 / 4 + quarterly * 0.5 / 12
    print(f"  weekly contacts: {weekly}, monthly: {monthly}, quarterly: {quarterly}")
    print(f"  estimated hours per week at 30 minutes per contact: {hours:.1f}")


if __name__ == "__main__":
    main()

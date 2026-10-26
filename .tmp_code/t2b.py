from datetime import date, timedelta

START = date(2026, 10, 26)
DECISION_DEADLINE = date(2027, 2, 1)

phases = [
    ("PhysioNet credentialing", 14),
    ("CITI training and certificate", 5),
    ("Project application and internal review", 10),
    ("Institutional sign-off", 10),
    ("Data access committee review", 14),
    ("Contract or DUA execution", 21),
    ("Transfer and checksum", 7),
]

day = START
print(f"Start: {START.isoformat()}   Deadline: {DECISION_DEADLINE.isoformat()}")
print(f"{'Phase':<44}{'Days':>6}{'Finish':>14}")
for name, days in phases:
    day = day + timedelta(days=days)
    print(f"{name:<44}{days:>6}{day.isoformat():>14}")
slack = (DECISION_DEADLINE - day).days
print()
print(f"Projected access date: {day.isoformat()}")
print(f"Slack against deadline: {slack} days")
if slack < 0:
    print("Verdict: access misses the decision window; trigger the fallback.")
elif slack < 14:
    print("Verdict: at risk; parallelise contract review with review.")
else:
    print("Verdict: feasible, with the stated assumptions.")

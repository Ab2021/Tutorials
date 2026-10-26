def epv(events, predictors):
    return events / predictors

scenarios = [
    ("All sites pooled", 18400, 0.11, 28),
    ("Largest site", 3100, 0.14, 28),
    ("Median site", 780, 0.09, 28),
    ("Smallest reporting site", 210, 0.08, 28),
]
print(f"{'Stratum':<26}{'Stays':>8}{'Prev':>7}{'Events':>8}{'EPV':>7}  Verdict")
for name, stays, prev, p in scenarios:
    events = round(stays * prev)
    value = epv(events, p)
    if value >= 10:
        verdict = "adequate"
    elif value >= 5:
        verdict = "marginal"
    else:
        verdict = "inadequate"
    print(f"{name:<26}{stays:>8}{prev:>7.0%}{events:>8}{value:>7.1f}  {verdict}")

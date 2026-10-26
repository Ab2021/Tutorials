import math

def lms_z(value, L, M, S):
    if L == 0:
        return math.log(value / M) / S
    return ((value / M) ** L - 1) / (L * S)

def pct(z):
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))

rows = [
    ("current weight, preterm 35+5", 2.04, -0.15, 2.40, 0.17),
    ("birth weight, 34+2", 1.98, -0.15, 2.10, 0.17),
    ("SGA birth weight, 34+2", 1.48, -0.15, 2.10, 0.17),
    ("keying error 20.4 kg", 20.4, -0.15, 2.40, 0.17),
    ("term chart at 10 days", 2.04, 0.20, 3.20, 0.13),
]
for label, v, L, M, S in rows:
    z = lms_z(v, L, M, S)
    print(f"{label}: z={z:+.2f}, percentile={100 * pct(z):.2f}")

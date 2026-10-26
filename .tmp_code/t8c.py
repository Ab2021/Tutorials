import math

def wilson(k, n, z=1.96):
    p = k / n
    denom = 1 + z*z/n
    centre = (p + z*z/(2*n)) / denom
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / denom
    return centre - half, centre + half

for k, n in ((320, 500), (160, 500), (48, 50)):
    lo, hi = wilson(k, n)
    print(f"{k}/{n} = {k/n:.0%}  95% CI {lo:.1%} to {hi:.1%}")
print()
print("A 64% pass rate on 500 items has a 95% interval of roughly 60% to 68%.")
print("Ranking two systems that differ by 2 points on 500 items is not supported.")

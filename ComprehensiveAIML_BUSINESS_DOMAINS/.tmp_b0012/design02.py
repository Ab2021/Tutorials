import math
import numpy as np

rng = np.random.default_rng(20261110)
lg = lambda x: math.log(x/(1-x))
sig = lambda z: 1/(1+math.exp(-z))

edges = [(0.00, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10), (0.10, 0.14),
         (0.14, 0.20), (0.20, 0.28), (0.28, 0.40), (0.40, 0.60), (0.60, 1.00)]
ns = [520, 400, 300, 230, 180, 130, 90, 60, 40, 50]
print("sum n", sum(ns))

# choose generative a,b so total positives ~180
best = None
for a100 in range(-40, 60):
    for b100 in range(80, 140):
        a, b = a100/100, b100/100
        ks = [round(ni*sig(a + b*lg((lo+hi)/2))) for ni, (lo, hi) in zip(ns, edges)]
        s = sum(ks)
        if best is None or abs(s-180) < abs(best[0]-180):
            best = (s, a, b, ks)
print("best", best)

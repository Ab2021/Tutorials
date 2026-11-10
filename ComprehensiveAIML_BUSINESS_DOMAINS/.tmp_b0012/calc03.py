import math
import numpy as np

# ECE with 5 merged bins vs 10 bins
bins = [(400, 0.0125, 0.01), (350, 0.02, 0.02), (300, 0.04, 0.04), (250, 0.056, 0.06),
        (200, 0.10, 0.08), (180, 20/180, 0.12), (140, 30/140, 0.18), (100, 0.30, 0.26),
        (60, 0.45, 0.40), (20, 0.75, 0.70)]
n = 2000
ece10 = sum(w/n*abs(o-p) for w, o, p in bins)
merged = []
for i in range(0, 10, 2):
    w1, o1, p1 = bins[i]; w2, o2, p2 = bins[i+1]
    W = w1+w2
    merged.append((W, (w1*o1+w2*o2)/W, (w1*p1+w2*p2)/W))
for m in merged:
    print("merged", m[0], round(m[1], 6), round(m[2], 6), round(abs(m[1]-m[2]), 6))
ece5 = sum(w/n*abs(o-p) for w, o, p in merged)
print("ECE10", round(ece10, 6), "ECE5", round(ece5, 6))

# logit prior shift .09 -> .12 and slope/intercept identity checks
def lg(p): return math.log(p/(1-p))
print("logit .09", lg(0.09), "logit .12", lg(0.12), "shift", lg(0.12)-lg(0.09))
print("logit .0824", lg(0.0824), "shift to .09", lg(0.09)-lg(0.0824))

# cost curve at thresholds
R, C, d = 18.0, 220.0, 0.55
print("breakeven p*", R/(d*C))
def cost(tau, verbose=False):
    flagged = [(w, o) for w, o, p in bins if p > tau]
    unf = [(w, o) for w, o, p in bins if p <= tau]
    nf = sum(w for w, o in flagged)
    ef = sum(w*o for w, o in flagged)
    eu = sum(w*o for w, o in unf)
    total = nf*R + (ef*(1-d) + eu)*C
    if verbose:
        print("tau", tau, "rev", nf, "ev flagged", round(ef, 2), "prevented", round(ef*d, 2),
              "ev unflagged", round(eu, 2), "cost", round(total, 2), "precision", round(ef/nf, 4))
    return total
for tau in [0.0, 0.0126, 0.0201, 0.0411, 0.0611, 0.0625, 0.0811, 0.1001, 0.1490, 0.1801, 0.2101, 0.2601, 0.3001, 0.4001, 0.7001]:
    cost(tau, True)
print("max cost all review", 2000*R + 180*(1-d)*C)
print("no review", 180*C)

# subgroup example in bin7
print("bin7 gap pooled", 30/140-0.18)
print("segA", 23/90-0.18, "segB", 7/50-0.18)
print("check", (23+7)/140)

# sample size for 1pp at p=0.2
for p in [0.02, 0.1, 0.3, 0.75]:
    print("p", p, "se at n=100", math.sqrt(p*(1-p)/100), "n for se 0.01", p*(1-p)/0.0001)

# NNR
print("NNR 0.0625", 700/142, "NNR 0.149", 320/102)
print("precision flagged", 142/700, "unflagged", 38/1300)
print("sensitivity at 0.0625", 142/180, "at 0.149", 102/180)
print("flag rate", 700/2000, 320/2000)

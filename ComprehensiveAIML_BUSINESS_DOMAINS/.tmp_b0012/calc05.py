import math
import numpy as np

p = [0.012, 0.022, 0.038, 0.055, 0.080, 0.110, 0.150, 0.200, 0.285, 0.430]
k = [3, 5, 8, 12, 17, 24, 31, 42, 60, 96]
n = 200
N = 2000
P = sum(k)
print("P", P, "base", P/N)
for pi, ki in zip(p, k):
    o = ki/n
    print(pi, o, round(o-pi, 4), "z", round((o-pi)/math.sqrt(pi*(1-pi)/n), 3))
ece = sum(abs(ki/n - pi) for pi, ki in zip(p, k))/10
print("ECE", round(ece, 5), "sum abs", round(sum(abs(ki/n-pi) for pi, ki in zip(p,k)),4))
base = P/N
rel = sum(((ki/n)-pi)**2 for pi, ki in zip(p, k))/10
res = sum((ki/n - base)**2 for ki in k)/10
unc = base*(1-base)
print("rel", round(rel,6), "res", round(res,6), "unc", round(unc,6), "BS", round(rel-res+unc,6))
bs = sum(pi*pi + ki/n - 2*pi*(ki/n) for pi, ki in zip(p, k))/10
print("BS direct", round(bs,6))
print("mean pred", round(sum(p)/10,4), "base", base)
print("logit shift", round(math.log(base/(1-base)) - math.log(0.1009/(1-0.1009)),4))
mp = sum(p)/10
print("mean pred exact", mp, "shift", round(math.log(base/(1-base))-math.log(mp/(1-mp)),4))
# AUC via bin ties
num = 0; neg_below = 0
for pi, ki in zip(p, k):
    pos = ki; neg = n - ki
    num += pos*(neg_below + 0.5*neg)
    neg_below += neg
print("AUC", round(num/(P*(N-P)), 4))
# bin edges implied by p
print("p list", p)
# thresholds
for tau in [0.05,0.10,0.15,0.25,0.30]:
    rev = sum(n for pi in p if pi > tau)
    ev = sum(ki for pi, ki in zip(p,k) if pi > tau)
    print("tau", tau, "reviews", rev, "events", ev, "prec", round(ev/rev,4) if rev else None,
          "sens", round(ev/P,4), "flagrate", rev/N)
# cost
R, C, d = 18.0, 220.0, 0.55
for tau in [0.05,0.10,0.15,0.25,0.30]:
    rev = sum(n for pi in p if pi > tau); ev = sum(ki for pi,ki in zip(p,k) if pi > tau)
    total = rev*R + (ev*(1-d) + (P-ev))*C
    print("cost", tau, total, "per item", round(total/N,3))
print("break-even", round(R/(d*C),4))

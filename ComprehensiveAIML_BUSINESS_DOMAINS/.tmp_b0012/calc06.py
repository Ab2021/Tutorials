import math
import numpy as np

p = [0.012, 0.022, 0.038, 0.055, 0.080, 0.110, 0.150, 0.200, 0.285, 0.430]
k = [3, 5, 8, 12, 17, 24, 31, 42, 60, 96]
n, N = 200, 2000
P = sum(k)

def wilson(kk, nn, z=1.96):
    ph = kk/nn
    d = 1 + z*z/nn
    c = (ph + z*z/(2*nn))/d
    h = z*math.sqrt(ph*(1-ph)/nn + z*z/(4*nn*nn))/d
    return (round(c-h, 4), round(c+h, 4))

print("tail bin 10 Wilson", wilson(96, 200), "pred 0.43")
print("tail bin 9 Wilson", wilson(60, 200), "pred 0.285")
print("bin 1 Wilson", wilson(3, 200))
# decision-weighted calibration error at tau=.25
flagged = [(pi, ki) for pi, ki in zip(p, k) if pi > 0.25]
dw = sum(abs(ki/n - pi)*n for pi, ki in flagged)/sum(n for pi, ki in flagged)
print("decision-weighted gap", round(dw, 4), "events", sum(ki for pi, ki in flagged))
overall = sum(abs(ki/n - pi) for pi, ki in zip(p, k))/10
print("overall ECE", round(overall, 5))

# intercept shift to new prevalence .19: solve for delta
base = P/N
tau_target = 0.19
c_now = math.log(base/(1-base))
c_new = math.log(tau_target/(1-tau_target))
delta = c_new - c_now
print("base", base, "delta logit", round(delta, 4), "odds ratio", round(math.exp(delta), 4))

# shifted scores and flagged counts at tau=.25 and tau=.15
def shift(pi, delta):
    return 1/(1+math.exp(-(math.log(pi/(1-pi)) + delta)))
shifted = [shift(pi, delta) for pi in p]
print("shifted p", [round(x, 4) for x in shifted])
for tau in [0.15, 0.25, 0.1488]:
    print("tau", tau, "flagged", sum(n for x in shifted if x > tau), "events", sum(ki for x, ki in zip(shifted, k) if x > tau))
# recalibrated with delta, at tau .25 how many flagged
# what tau on old scale gives same 400
print("tau to keep 400 with shift: between bin8/.9")

# decision rule under capacity 450
R, C, d = 18.0, 220.0, 0.55
def cost(tau):
    rev = sum(n for pi in p if pi > tau); ev = sum(ki for pi, ki in zip(p, k) if pi > tau)
    return rev*R + (ev*(1-d) + (P-ev))*C, rev
for tau in [0.15, 0.25]:
    c, rev = cost(tau)
    print("tau", tau, "cost", c, "reviews", rev, "per item", round(c/N, 3))
print("cost of binding constraint", cost(0.25)[0]-cost(0.15)[0], round((cost(0.25)[0]-cost(0.15)[0])/cost(0.15)[0], 4))

# capacity in hours
print("450 reviews * 6 min =", 450*6, "min =", 450*6/60, "h")

# sample size for tail precision
for kk, nn_ in [(96, 200), (240, 800), (480, 1600)]:
    ph = kk/nn_
    print("n", nn_, "se", round(math.sqrt(ph*(1-ph)/nn_), 4))

# Brier decomposition check
rel = sum(((ki/n)-pi)**2 for pi, ki in zip(p, k))/10
res = sum((ki/n - base)**2 for ki in k)/10
unc = base*(1-base)
print("rel", round(rel, 7), "res", round(res, 6), "unc", round(unc, 6), "sum", round(rel-res+unc, 6))
bs = sum(pi*pi + ki/n - 2*pi*(ki/n) for pi, ki in zip(p, k))/10
print("BS", round(bs, 6))

# NNR
print("NNR tau .25", round(N/400, 3), "sens .5235", "prec", round(156/400, 4))

# expected cost of one review vs expected prevented loss at p
for pp in [0.15, 0.25, 0.39, 0.48]:
    print("p", pp, "expected prevented loss", round(pp*d*C, 2), "review cost", R, "net", round(pp*d*C-R, 2))

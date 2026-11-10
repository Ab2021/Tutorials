import math
import numpy as np

p = np.array([.012, .026, .045, .070, .100, .135, .175, .235, .320, .470])
k = np.array([2, 5, 8, 13, 19, 26, 33, 42, 52, 68])
n = 200.
N = 2000
P = k.sum()
o = k/n
print("P", P, "base", P/N, "meanp", p.mean())
print("ECE", np.mean(np.abs(o-p)))
print("gaps", np.round(o-p, 4))
base = P/N
print("logit base", math.log(base/(1-base)), "logit meanp", math.log(p.mean()/(1-p.mean())))
print("delta cilt", math.log(p.mean()/(1-p.mean())) - math.log(base/(1-base)))

# binomial Newton fit logit(o) = a + b logit(p)
lp = np.log(p/(1-p))
a, b = 0.0, 1.0
for it in range(200):
    eta = a + b*lp
    mu = 1/(1+np.exp(-eta))
    g = np.array([np.sum(k - n*mu), np.sum((k - n*mu)*lp)])
    wgt = n*mu*(1-mu)
    H = np.array([[wgt.sum(), (wgt*lp).sum()], [(wgt*lp).sum(), (wgt*lp*lp).sum()]])
    step = np.linalg.solve(H, g)
    a += step[0]; b += step[1]
    if np.max(np.abs(step)) < 1e-12:
        break
print("fit a,b", round(a,5), round(b,5), "iters", it)
recal = 1/(1+np.exp(-(a+b*lp)))
print("recal", np.round(recal,4))
print("ECE recal", np.mean(np.abs(o-recal)))
print("ECE recal weighted", np.sum(np.abs(k-n*recal))/N)
bs_new = np.mean(recal**2 + o - 2*recal*o)
print("BS after", round(bs_new,6))

# Brier decomposition
rel = np.mean((o-p)**2); res = np.mean((o-base)**2); unc = base*(1-base)
print("rel res unc sum", round(rel,6), round(res,6), round(unc,6), round(rel-res+unc,6))
bs = np.mean(p**2 + o - 2*p*o)
print("BS", round(bs,6))
print("skill vs climatology", round(1-bs/unc,4))
print("rel/unc", round(rel/unc,4))

# sign-invariance demo: flip bin3 residual
p2 = p.copy(); p2[2] = o[2] + 0.005  # predicted 0.045 -> 0.045 (gap +0.005)
p2[3] = o[3] + 0.005  # 0.065 -> 0.070? gap +0.005
print("variant gaps", np.round(o-p2,4))
print("ECE equal-count variant", np.mean(np.abs(o-p2)))
merged = [ (p2[2]*n+p2[3]*n)/(2*n), (p2[4]*n+p2[5]*n)/(2*n) ]
print("merged pairs 2-3 mean pred", round((o[2]+o[3])/2,4), "mean p", round((p2[2]+p2[3])/2,4))
print("merged gap", round((o[2]+o[3])/2 - (p2[2]+p2[3])/2,4))

# Wilson for key bins
def wil(kk, nn, z=1.96):
    ph = kk/nn; d = 1+z*z/nn
    c = (ph+z*z/(2*nn))/d
    h = z*math.sqrt(ph*(1-ph)/nn + z*z/(4*nn*nn))/d
    return round(c-h,4), round(c+h,4)
for kk in [2, 13, 52, 68]:
    print("wilson", kk, wil(kk, 200))

# cost model
R, C, d = 18.0, 220.0, 0.55
print("pstar", R/(d*C))
def total(tau, d_=d):
    flag = p > tau
    rev = n*flag.sum(); ev = k[flag].sum(); miss = P-ev
    return rev*R + (miss + (1-d_)*ev)*C, rev, ev
for tau in [-1, .10, .20, .30, .45]:
    t, rev, ev = total(tau)
    print("tau", tau, "reviews", rev, "events", ev, "cost", t, "prec", round(ev/rev,4) if rev else 0,
          "sens", round(ev/P,4))
print("flag none", P*C)
# d decay
for dd in [0.55, 0.40, 0.20, 0.10]:
    print("d", dd, "pstar", round(R/(dd*C),4))
# Little's law growth
arr, cap = 600, 400
for wk in [1, 2, 4, 8, 13, 26]:
    backlog = (arr-cap)*wk
    print("week", wk, "backlog", backlog, "wait wk", round(backlog/cap,3), "wait days", round(backlog/cap*7,2))
# prior shift
newb = 0.19
delta = math.log(newb/(1-newb)) - math.log(base/(1-base))
print("delta shift", round(delta,4), "odds mult", round(math.exp(delta),4))
ps = 1/(1+np.exp(-(lp+delta)))
print("shifted p", np.round(ps,4))
for tau in [.20, .30]:
    print("tau", tau, "flagged", n*(ps > tau).sum(), "events", k[ps > tau].sum())
print("expected events from model", 2000*p.mean(), "observed", P)
# queue at 0.20 rule with capacity 400
print("overrun", 600-400, "hours", (600-400)*6/60)

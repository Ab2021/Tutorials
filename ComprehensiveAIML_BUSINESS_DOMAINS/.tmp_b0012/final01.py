import math
import numpy as np

p = np.array([.012, .026, .045, .070, .100, .135, .175, .235, .320, .470])
k = np.array([2, 5, 8, 13, 19, 26, 33, 42, 52, 68])
n = 200.0
N = 2000
P = k.sum()
base = P/N
o = k/n
print("P", P, "base", round(base, 6), "meanp", round(p.mean(), 6))
print("CIL", round(p.mean()-base, 6))
print("ECE", round(np.mean(np.abs(o-p)), 6))
print("all gaps negative:", bool(np.all(o-p < 0)))

# merged pairs ECE
merges = []
for i in range(0, 10, 2):
    W = 400.0
    merges.append(((k[i]+k[i+1])/W, (p[i]+p[i+1])/2))
ece5 = sum(abs(a-b) for a, b in merges)/5
print("pairs", [(round(a,4), round(b,4), round(abs(a-b),4)) for a,b in merges], "ECE5", round(ece5,6))

# Wilson per bin
def wil(kk, nn, z=1.96):
    ph = kk/nn; d = 1+z*z/nn
    c = (ph+z*z/(2*nn))/d
    h = z*math.sqrt(ph*(1-ph)/nn+z*z/(4*nn*nn))/d
    return round(c-h, 4), round(c+h, 4)
for kk in k:
    print("wilson", int(kk), wil(kk, 200))

# Brier
rel = np.mean((o-p)**2); res = np.mean((o-base)**2); unc = base*(1-base)
bs = np.mean(p**2 + o - 2*p*o)
print("rel", round(rel,7), "res", round(res,7), "unc", round(unc,7))
print("sum", round(rel-res+unc,7), "bs", round(bs,7), "skill", round(1-bs/unc,6))

def ll(pp):
    pp = np.clip(pp, 1e-12, 1-1e-12)
    return -np.mean(o*np.log(pp)+(1-o)*np.log(1-pp))
print("ll model", round(ll(p),6), "ll clim", round(ll(np.full(10, base)),6))

# AUROC tie corrected
num = 0.0; below = 0.0
for pi, ki in zip(p, k):
    pos = ki; neg = n-ki
    num += pos*(below+0.5*neg); below += neg
print("auroc", round(num/(P*(N-P)),6))

# cost
R, C, d = 18.0, 220.0, 0.55
print("pstar", round(R/(d*C),6))
def cost(tau):
    flag = p > tau
    rev = n*flag.sum(); ev = k[flag].sum(); miss = P-ev
    return rev*R + (miss+(1-d)*ev)*C, rev, ev, miss
for tau in [0.15, 0.20, 0.30]:
    c, rev, ev, miss = cost(tau)
    print("tau", tau, "rev", int(rev), "ev", int(ev), "missed", int(miss),
          "cost", c, "per item", round(c/N,3), "prec", round(ev/rev,4), "sens", round(ev/P,4))
print("flag none", P*C, "flag all", N*R + (1-d)*P*C)
c15 = cost(0.15)[0]; c20 = cost(0.20)[0]; c30 = cost(0.30)[0]
print("saving 0.15 vs none", P*C-c15, "0.20 vs none", P*C-c20, "0.30 vs none", P*C-c30)
print("capacity cost", c30-c15, "share of saving", round((c30-c15)/(P*C-c15),4))
print("incumbent vs feasible", c30-c20)

# queue
cap = 450.0
for dem in [600.0, 800.0]:
    print("demand", dem, "hours", dem*6/60, "overrun/wk", dem-cap)
    for wk in [1, 4, 13]:
        bl = (dem-cap)*wk
        print("  wk", wk, "backlog", bl, "wait wk", round(bl/cap,3), "days", round(bl/cap*7,1))

# prior shift
new = 0.19
delta = math.log(new/(1-new)) - math.log(base/(1-base))
print("delta", round(delta,6), "odds mult", round(math.exp(delta),5))
lp = np.log(p/(1-p))
ps = 1/(1+np.exp(-(lp+delta)))
print("shifted", np.round(ps,4))
for tau in [0.20, 0.30]:
    f = ps > tau
    print("tau", tau, "rev", int(n*f.sum()), "ev", int(k[f].sum()))
print("rel prevalence increase", round(new/base-1,4))

# Platt fit
a, b = 0.0, 1.0
for it in range(200):
    mu = 1/(1+np.exp(-(a+b*lp)))
    g = np.array([np.sum(k-n*mu), np.sum((k-n*mu)*lp)])
    wg = n*mu*(1-mu)
    H = np.array([[wg.sum(), (wg*lp).sum()], [(wg*lp).sum(), (wg*lp*lp).sum()]])
    step = np.linalg.solve(H, g); a += step[0]; b += step[1]
    if np.abs(step).max() < 1e-12: break
print("platt", round(a,6), round(b,6), "iters", it)
rc = 1/(1+np.exp(-(a+b*lp)))
print("recal", np.round(rc,4))
print("ECE recal", round(np.mean(np.abs(o-rc)),6), "CIL recal", round(rc.mean()-base,6))
print("ll recal", round(ll(rc),6))

# decision-weighted gap at tau .30
f = p > 0.30
print("DW gap .30", round(np.sum(np.abs(o[f]-p[f])*n)/np.sum(n*f),4))
print("avg p flagged", round((p[f]*n).sum()/(n*f.sum()),4), "obs", round(k[f].sum()/(n*f.sum()),4))

# sample size
for pi in [0.175, 0.47]:
    for se in [0.01, 0.02]:
        print("p", pi, "se", se, "n", math.ceil(pi*(1-pi)/se**2))

# NNR
print("NNR .30", round(N/400,3), "NNR .20", round(N/600,3))

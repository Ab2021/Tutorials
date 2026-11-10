import math
import numpy as np

# ---------- case 01 extras ----------
p = np.array([.012, .026, .045, .070, .100, .135, .175, .235, .320, .470])
k = np.array([2, 5, 8, 13, 19, 26, 33, 42, 52, 68])
n = 200.; N = 2000; P = k.sum(); base = P/N
o = k/n
print("base", base)

# AUROC with tie correction (bin-constant scores)
num = 0.0; neg_below = 0.0
for pi, ki in zip(p, k):
    pos = ki; neg = n - ki
    num += pos*(neg_below + 0.5*neg)
    neg_below += neg
print("AUROC", round(num/(P*(N-P)), 4))

# log loss
def ll(pp):
    pp = np.clip(pp, 1e-12, 1-1e-12)
    return -np.mean(o*np.log(pp) + (1-o)*np.log(1-pp))
print("logloss base", round(ll(p), 5), "climatology", round(ll(np.full(10, base)), 5))
a, b = -0.43079, 0.84472
lp = np.log(p/(1-p))
rc = 1/(1+np.exp(-(a+b*lp)))
print("logloss recal", round(ll(rc), 5))

# binomial SE per bin
for pi, ki in zip(p, k):
    print("bin p", pi, "se", round(math.sqrt(pi*(1-pi)/n), 4))

# sample size rules
for pi in [0.05, 0.235, 0.47]:
    for se in [0.01, 0.02, 0.03]:
        print("p", pi, "se", se, "n", math.ceil(pi*(1-pi)/se**2))

# cost tables for d variants
R, C = 18.0, 220.0
def cost(d, tau):
    flag = p > tau
    rev = n*flag.sum(); ev = k[flag].sum()
    return rev*R + ((P-ev) + (1-d)*ev)*C, rev, ev
for d in [0.55, 0.40, 0.20]:
    print("d", d, "pstar", round(R/(d*C), 4))
    for tau in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        c, rev, ev = cost(d, tau)
        print("   tau", tau, "rev", rev, "ev", ev, "cost", round(c, 1), "per item", round(c/N, 3))

# capacity 450/week, 6 min per review
cap, dem = 450, 600
print("hours demanded", dem*6/60, "capacity hours", cap*6/60, "over h", (dem-cap)*6/60)
for wk in [1, 2, 4, 13, 26, 52]:
    backlog = (dem-cap)*wk
    print("wk", wk, "backlog", backlog, "wait wk", round(backlog/cap, 2), "days", round(backlog/cap*7, 1))

# prior shift
new = 0.19
delta = math.log(new/(1-new)) - math.log(base/(1-base))
ps = 1/(1+np.exp(-(lp+delta)))
print("delta", round(delta, 4), "odds", round(math.exp(delta), 4))
print("shift p", np.round(ps, 4))
for tau in [0.20, 0.30, 0.35]:
    flag = ps > tau
    print("tau", tau, "rev", n*flag.sum(), "ev", k[flag].sum())

# NNR at tau .20
print("NNR", round(N/600, 3), "sens", round(162/P, 4), "prec", round(162/600, 4))
print("missing events", P-162, "missed cost", (P-162)*C, "review cost", 600*R, "total", (P-162)*C+600*R)

# ---------- case 02 extras ----------
x = np.array([1., 2., 3., 4., 5.]); y = np.array([1.9, 3.1, 4.2, 5.6, 6.2])
X = np.column_stack([np.ones(5), x])
H = (2/5)*(X.T@X)
ev = np.linalg.eigvalsh(H); L, mu = ev.max(), ev.min()
print("L mu kappa", L, mu, L/mu)
for lam in [0.5, 1.0, 2.0]:
    Hr = H + 2*lam*np.eye(2)
    e = np.linalg.eigvalsh(Hr); kap = e.max()/e.min()
    print("ridge", lam, "kappa", round(kap,4), "iters", round(math.log(1e-3)/math.log(1-1/kap),2),
          "mom iters", round(math.log(1e-3)/math.log((math.sqrt(kap)-1)/(math.sqrt(kap)+1)),2))
# Armijo backtracking from w=0
def f(w): return np.mean((X@w-y)**2)
w0 = np.zeros(2); g0 = (2/5)*X.T@(X@w0-y)
print("f0", round(f(w0), 4), "gnorm2", round(g0@g0, 4))
eta = 1.0; c = 1e-4
for step in range(40):
    lhs = f(w0 - eta*g0); rhs = f(w0) - c*eta*(g0@g0)
    print("eta", round(eta,6), "lhs", round(lhs,4), "rhs", round(rhs,4), "ok", lhs <= rhs)
    if lhs <= rhs: break
    eta /= 2
# optimal-step GD trace
eta_opt = 2/(L+mu)
print("eta_opt", round(eta_opt, 6))
w = np.zeros(2)
for kk in range(1, 6):
    g = (2/5)*X.T@(X@w-y)
    w = w - eta_opt*g
    print("k", kk, "w", np.round(w, 4), "loss", round(f(w), 6))

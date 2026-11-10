import math
import numpy as np

x = np.array([1., 2., 3., 4., 5.]); y = np.array([1.9, 3.1, 4.2, 5.6, 6.2])
n = 5
X = np.column_stack([np.ones(n), x])
H = (2/n)*(X.T@X)
ev = np.linalg.eigvalsh(H); L, mu = ev.max(), ev.min(); kap = L/mu
print("L", L, "mu", mu, "kappa", kap)
rho = (kap-1)/(kap+1)
print("rho opt", rho, "iters to 1e-3", math.log(1e-3)/math.log(rho))
print("rho at 1/L", 1-1/kap, "iters", math.log(1e-3)/math.log(1-1/kap))
print("eta 1/L", 1/L, "eta 2/L", 2/L, "eta opt", 2/(L+mu))
# verify factor at eta
for eta in [1/L, 2/(L+mu), 2/L, 2/L*1.02]:
    print("eta", round(eta,6), "max|1-eta lam|", round(max(abs(1-eta*ev[0]), abs(1-eta*ev[1])),6))
# exact iterations by simulation at eta=2/(L+mu)
w = np.zeros(2); wstar = np.linalg.solve(X.T@X, X.T@y)
dist = []
for k in range(400):
    w = w - (2/(L+mu))*(2/n)*X.T@(X@w-y)
    dist.append(np.linalg.norm(w-wstar))
import bisect
for target in [1e-1, 1e-2, 1e-3]:
    k = next(i for i, dd in enumerate(dist) if dd < target)
    print("dist<", target, "at iter", k)
print("dist ratio last", dist[100]/dist[99], "theory", rho)
# eta=1/L iterations
w = np.zeros(2); cnt = {}
for k in range(1, 900):
    w = w - (1/L)*(2/n)*X.T@(X@w-y)
    d = np.linalg.norm(w-wstar)
    for t in [1e-1, 1e-2, 1e-3]:
        if d < t and t not in cnt: cnt[t] = k
print("eta=1/L iters", cnt)
# momentum heavy ball: x_{k+1} = x_k - eta grad + beta(x_k - x_{k-1})
beta = ((math.sqrt(kap)-1)/(math.sqrt(kap)+1))**2
print("beta", beta, "sqrt beta", math.sqrt(beta))
eta_m = 4/(math.sqrt(L)+math.sqrt(mu))**2
print("eta_m", eta_m)
w = np.zeros(2); wp = np.zeros(2); cnt = {}
for k in range(1, 200):
    g = (2/n)*X.T@(X@w-y)
    wn = w - eta_m*g + beta*(w-wp)
    wp, w = w, wn
    d = np.linalg.norm(w-wstar)
    for t in [1e-3]:
        if d < t and t not in cnt: cnt[t] = k
print("momentum iters to 1e-3", cnt)
# ridge with standardized features
# scaled feature: s = (x-3)/1.4142
s = (x - x.mean())/np.std(x)
Xs = np.column_stack([np.ones(n), s])
Hs = (2/n)*(Xs.T@Xs)
print("Hs", Hs.tolist(), "evals", np.linalg.eigvalsh(Hs))
# flops
for nrow, d_ in [(200000, 12), (2000000, 12)]:
    print("n", nrow, "d", d_, "flops/iter", 4*nrow*d_, "MB", nrow*d_*8/1e6)
print("SGD epoch flops", 4*200000*12)
print("full 242 iters", 242*4*200000*12, "SGD 20 epochs", 20*4*200000*12,
      "ratio", 242/20)
# variance of minibatch gradient
print("grad noise var factor 1/b", 1/1, 1/64)
# duplicate column gradient trajectory
x2 = np.column_stack([x, 2*x])
print("pinv", np.linalg.pinv(x2)@y)
print("loss at min norm", np.mean((x2@(np.linalg.pinv(x2)@y)-y)**2))
print("loss at (0.87, 0) etc:", np.mean((x2@np.array([0.87,0.0])-y)**2))

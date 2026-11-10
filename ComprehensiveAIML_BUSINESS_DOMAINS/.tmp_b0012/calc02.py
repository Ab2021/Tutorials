import numpy as np
np.set_printoptions(precision=6, suppress=True)

x = np.array([1., 2., 3., 4., 5.])
y = np.array([1.9, 3.1, 4.2, 5.6, 6.2])
n = len(x)
X = np.column_stack([np.ones(n), x])
XtX = X.T @ X
Xty = X.T @ y
print("XtX", XtX.tolist())
print("Xty", Xty.tolist())
w = np.linalg.solve(XtX, Xty)
print("w_ols", w)
H = (2/n) * XtX
print("H", H.tolist())
ev = np.linalg.eigvalsh(H)
print("evals H", ev, "L", ev.max(), "kappa", ev.max()/ev.min())
print("eta max 2/L", 2/ev.max())
# residual, loss at optimum
r = X @ w - y
print("resid", r, "SSE", r @ r, "loss", (r@r)/n)
# gradient at 0
print("grad at 0", (-2/n) * Xty)
eta = 0.005
wk = np.zeros(2)
for k in range(1, 6):
    g = (2/n) * X.T @ (X @ wk - y)
    wk = wk - eta*g
    f = ((X@wk - y)**2).sum()/n
    print("iter", k, wk, "loss", round(f, 8))
# contraction factors
for eta_t in [0.005, 0.01, 2/ev.max(), 2/ev.max()*1.02]:
    fac = max(abs(1-eta_t*ev.max()), abs(1-eta_t*ev.min()))
    print("eta", round(eta_t,6), "factor", round(fac,6))
# iterations to 1e-3 with optimal step 2/(L+mu)
L, mu = ev.max(), ev.min()
kappa = L/mu
eta_opt = 2/(L+mu)
fac = 1 - 1/kappa
import math
print("eta_opt", eta_opt, "fac", fac, "iters 1e-3", math.log(1e-3)/math.log(fac))
# centered
xc = x - x.mean()
Xc = np.column_stack([np.ones(n), xc])
Hc = (2/n) * Xc.T @ Xc
evc = np.linalg.eigvalsh(Hc)
print("Hc", Hc.tolist(), "evals", evc, "kappa", evc.max()/evc.min())
print("iters centered", math.log(1e-3)/math.log(1-1/(evc.max()/evc.min())))
# ridge
for lam in [0.5, 1.0, 5.0]:
    Hr = H + 2*lam*np.eye(2)
    e = np.linalg.eigvalsh(Hr)
    print("lam", lam, "kappa", e.max()/e.min(), "w_ridge", np.linalg.solve(XtX + n*lam*np.eye(2), Xty))
# GD with momentum-free vs momentum
# collinearity
X2 = np.column_stack([x, 2*x])
print("rank", np.linalg.matrix_rank(X2.T@X2))
H2 = (2/n) * (X2.T @ X2)
print("H2 evals", np.linalg.eigvalsh(H2))
# GD from zero on singular system, 200 iters, small step
eta2 = 0.001
w2 = np.zeros(2)
for k in range(4000):
    g = (2/n) * X2.T @ (X2 @ w2 - y)
    w2 = w2 - eta2*g
print("w2 after 4000", w2, "min-norm check", np.linalg.pinv(X2) @ y)

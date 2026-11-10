import math
import numpy as np

bins = [(400, 0.0125, 0.01), (350, 0.02, 0.02), (300, 0.04, 0.04), (250, 0.056, 0.06),
        (200, 0.10, 0.08), (180, 20/180, 0.12), (140, 30/140, 0.18), (100, 0.30, 0.26),
        (60, 0.45, 0.40), (20, 0.75, 0.70)]
n = 2000
P = 180

def wilson(k, n_, z=1.96):
    p = k/n_
    d = 1 + z*z/n_
    c = (p + z*z/(2*n_))/d
    h = z*math.sqrt(p*(1-p)/n_ + z*z/(4*n_*n_))/d
    return c-h, c+h

print("Wilson 30/140", [round(x,4) for x in wilson(30,140)])
print("Wilson 15/20", [round(x,4) for x in wilson(15,20)])
print("Wilson 5/400", [round(x,4) for x in wilson(5,400)])
print("Wilson 18/180", [round(x,4) for x in wilson(18,180)])
info = sum(w*p*(1-p) for w,o,p in bins)
print("sum n p(1-p)", round(info,3), "SE intercept", round(1/math.sqrt(info),4))
print("SE intercept per 1000 items", round(math.sqrt(1000/info*info)/1,4))
# cost per item
for tau in [0.0625, 0.149]:
    flagged = [(w,o) for w,o,p in bins if p > tau]
    unf = [(w,o) for w,o,p in bins if p <= tau]
    nf = sum(w for w,o in flagged); ef = sum(w*o for w,o in flagged); eu = sum(w*o for w,o in unf)
    total = nf*18 + (ef*0.45+eu)*220
    print("tau", tau, "total", total, "per item", round(total/n,3), "reviews", nf)
# capacity constraint 350
print("capacity 350: tau .149 uses 320 ok; tau .0811 uses 500 -> infeasible")
# breakeven sensitivity
for R,C,d in [(18,220,0.55),(36,220,0.55),(18,110,0.55),(18,220,0.35),(9,220,0.55)]:
    print("R,C,d",R,C,d,"p*",round(R/(d*C),4))
# momentum / GD numbers for case 02
L, mu = 23.661903789690598, 0.338096
kappa = L/mu
print("sqrt kappa", round(math.sqrt(kappa),4))
print("heavy ball rate", round((math.sqrt(kappa)-1)/(math.sqrt(kappa)+1),4))
rate = (math.sqrt(kappa)-1)/(math.sqrt(kappa)+1)
print("iters 1e-3 momentum", round(math.log(1e-3)/math.log(rate),2))
print("GD optimal rate", round(1-1/kappa,5), "iters", round(math.log(1e-3)/math.log(1-1/kappa),2))
print("kappa X", round(math.sqrt(kappa),4), "digits lost log10", round(math.log10(kappa),3), "double digits", round(math.log10(kappa*math.sqrt(2**-52)),3))
print("float64 eps digits", round(-math.log10(2**-52),2))
# flops
nrow, d = 200000, 12
fb = 4*nrow*d
print("full batch flops/iter", fb, "480 iters", fb*480, "SGD 20 epochs", 4*d*nrow*20, "ratio", round(fb*480/(4*d*nrow*20),1))
print("memory MB", nrow*d*8/1e6)
# scale blowup
x = np.linspace(0.01, 1, 200)
X = np.column_stack([np.ones_like(x), x])
print("kappa small scale", round(np.linalg.cond(X),2))
x2 = x*1e6
X2 = np.column_stack([np.ones_like(x2), x2])
print("kappa big scale", round(np.linalg.cond(X2),2))
# ridge shrinkage formula check: w(lam) components
XtX = np.array([[5.,15.],[15.,55.]]); Xty=np.array([21.,74.1])
for lam in [0,0.5,1,5]:
    w = np.linalg.solve(XtX + 5*lam*np.eye(2), Xty)
    print("lam", lam, w, "norm", round(np.linalg.norm(w),4))
# eigenvector of min eigenvalue
H = (2/5)*XtX
ev, evec = np.linalg.eigh(H)
print("evals", ev, "min evec", evec[:,0], "max evec", evec[:,1])
# gradient norm trajectory at eta .005
w = np.zeros(2); y=np.array([1.9,3.1,4.2,5.6,6.2]); Xd=np.column_stack([np.ones(5),np.array([1.,2,3,4,5])])
for k in range(1,9):
    g=(2/5)*Xd.T@(Xd@w-y); w = w-0.005*g
    print("k",k,"gnorm",round(np.linalg.norm(g),4),"loss",round(((Xd@w-y)**2).sum()/5,6))
# distance to optimum
wstar=np.array([0.87,1.11])
print("||w0-w*||", round(np.linalg.norm(wstar),4))
print("check contraction at .005: max|1-eta*L|", round(abs(1-0.005*L),5))

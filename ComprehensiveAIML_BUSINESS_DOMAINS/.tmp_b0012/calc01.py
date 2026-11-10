import math

bins = [(400, 0.0125, 0.01), (350, 0.02, 0.02), (300, 0.04, 0.04), (250, 0.056, 0.06),
        (200, 0.10, 0.08), (180, 20/180, 0.12), (140, 30/140, 0.18), (100, 0.30, 0.26),
        (60, 0.45, 0.40), (20, 0.75, 0.70)]
n = 2000
P = 180
meanp = sum(w*p for w, o, p in bins)/n
print("meanp", meanp)
print("intercept shift (logit obs - logit meanpred)", math.log(0.09/0.91) - math.log(meanp/(1-meanp)))
for w, o, p in bins:
    se = math.sqrt(p*(1-p)/w)
    print("n", w, "obs", round(o, 6), "pred", p, "gap", round(o-p, 5), "se", round(se, 5), "z", round((o-p)/se, 3))
neg_below = 0
num = 0
for w, o, p in bins:
    pos = w*o
    neg = w-pos
    num += pos*(neg_below + 0.5*neg)
    neg_below += neg
print("AUCbin", num/(P*(n-P)))
lp = [math.log(p/(1-p)) for w, o, p in bins]
lo = [math.log(o/(1-o)) for w, o, p in bins]
print("logit pred", [round(x, 4) for x in lp])
print("logit obs", [round(x, 4) for x in lo])
# weighted least squares of logit(obs) on logit(pred)
sxx = sum(w*(x**2) for (w, o, p), x in zip(bins, lp))
sx = sum(w*x for (w, o, p), x in zip(bins, lp))
sy = sum(w*y for (w, o, p), y in zip(bins, lo))
sxy = sum(w*x*y for (w, o, p), x, y in zip(bins, lp, lo))
det = n*sxx - sx*sx
b = (n*sxy - sx*sy)/det
a = (sy - b*sx)/n
print("WLS logit recal a,b", a, b)
# binomial Newton recalibration
a, b = 0.0, 1.0
for it in range(60):
    mu = [1/(1+math.exp(-(a+b*x))) for x in lp]
    g1 = sum(w*(o - m) for (w, o, p), m in zip(bins, mu))
    g2 = sum(w*(o - m)*x for (w, o, p), m, x in zip(bins, mu, lp))
    h11 = sum(w*m*(1-m) for (w, o, p), m in zip(bins, mu))
    h12 = sum(w*m*(1-m)*x for (w, o, p), m, x in zip(bins, mu, lp))
    h22 = sum(w*m*(1-m)*x*x for (w, o, p), m, x in zip(bins, mu, lp))
    det = h11*h22 - h12*h12
    da = (h22*g1 - h12*g2)/det
    db = (h11*g2 - h12*g1)/det
    a += da
    b += db
print("binomial recal a,b", a, b)
# ECE after Platt shift 0.03 -> 0.09 style: apply a+b*logit
def sig(z):
    return 1/(1+math.exp(-z))
ece = 0
for (w, o, p), x in zip(bins, lp):
    q = sig(a+b*x)
    ece += w/n*abs(o-q)
print("ECE after recal", ece)
# Brier after
bs = sum(w/n*(sig(a+b*x)**2 + o - 2*sig(a+b*x)*o) for (w, o, p), x in zip(bins, lp))
print("BS after", bs)
for (w, o, p), x in zip(bins, lp):
    q = sig(a+b*x)
    print("bin", w, "obs", round(o, 6), "recal", round(q, 5))
print("SE p=0.3 n=100", math.sqrt(.3*.7/100))
print("n needed se .005 at p=.0564", .0564/0.000025)
print("review cost per item", 12/192)
print("precision at 0.0625 flag", 0.09/0.35)

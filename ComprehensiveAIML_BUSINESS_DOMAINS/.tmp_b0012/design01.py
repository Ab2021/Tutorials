import math

w = [400, 350, 300, 250, 200, 180, 140, 100, 60, 20]
p = [0.01, 0.02, 0.04, 0.06, 0.08, 0.12, 0.18, 0.26, 0.40, 0.70]
lg = lambda x: math.log(x/(1-x))

def counts(c, slope=0.75):
    out = []
    for wi, pi in zip(w, p):
        o = 1/(1+math.exp(-(c + slope*lg(pi))))
        out.append(round(wi*o))
    return out

best = None
for i in range(-4000, 2000):
    c = i/1000
    k = counts(c)
    s = sum(k)
    if best is None or abs(s-180) < abs(best[1]-180):
        best = (c, s, k)
print("best", best)

for c in [x/100 for x in range(-60, 10)]:
    k = counts(c)
    print(round(c,2), sum(k), k)

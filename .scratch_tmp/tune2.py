import math, random, statistics
from tune import build, quantile

SEED = 20261027
obs, sim = build()

def make_prob(sizes, target, klo, khi, kcalc, b0):
    def one(s, d, c):
        o = (s - d) / d
        if o < 0.0 or o > 0.45:
            return 0.0
        pen = klo * (target - o) ** 2 if o < target else khi * (o - target) ** 2
        return 1 / (1 + math.exp(-(b0 - pen - kcalc * c)))
    def p(d, c):
        return max(one(s, d, c) for s in sizes)
    return p

def report(p):
    ps = sum(p(d, c) for d, c in sim) / len(sim)
    po = sum(p(d, c) for d, c in obs) / len(obs)
    ys = sorted(obs, key=lambda t: t[0]); step = len(ys) // 10
    dec = []
    for i in range(10):
        ch = ys[i*step:(i+1)*step] if i < 9 else ys[9*step:]
        dec.append((sum(p(d,c) for d,c in ch)/len(ch), statistics.mean(c for _,c in ch)))
    # decomposition at fixed reference values
    geom = sum(p(d, 0.30) for d, _ in sim)/len(sim) - sum(p(d, 0.30) for d, _ in obs)/len(obs)
    calc = sum(p(24.5, c) for _, c in sim)/len(sim) - sum(p(24.5, c) for _, c in obs)/len(obs)
    return ps, po, ps-po, dec, geom, calc

for sizes in [(20,22,24,26,28,30)]:
    for args in [(0.13,1200,25,4.0,4.6),(0.13,1200,25,3.0,4.2),(0.12,1500,30,3.5,4.4),(0.14,1000,20,3.5,4.5)]:
        p = make_prob(sizes, *args)
        ps, po, diff, dec, geom, calc = report(p)
        print(f"{args} ps={ps:.4f} po={po:.4f} opt={diff:+.4f} geom={geom:+.4f} calc={calc:+.4f} sum={geom+calc:+.4f}")
        print("   deciles:", " ".join(f"{v:.3f}" for v, _ in dec))
        print("   calc   :", " ".join(f"{c:.2f}" for _, c in dec))

import math, random, statistics

SEED, SIZES = 20261027, (20.0, 22.0, 24.0, 26.0, 28.0, 30.0)


def q(xs, p):
    ys = sorted(xs); pos = p * (len(ys) - 1); lo = int(pos)
    return ys[lo] + (pos - lo) * (ys[min(lo + 1, len(ys) - 1)] - ys[lo])


def ks(a, b):
    sa, sb = sorted(a), sorted(b); ia = ib = 0; best = 0.0
    for x in sorted(set(a) | set(b)):
        while ia < len(sa) and sa[ia] <= x: ia += 1
        while ib < len(sb) and sb[ib] <= x: ib += 1
        best = max(best, abs(ia / len(sa) - ib / len(sb)))
    return best


def smd(ma, sa, mb, sb):
    return (ma - mb) / math.sqrt((sa * sa + sb * sb) / 2)


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def cohorts(lim=2.0, cap=0.72):
    rng = random.Random(SEED); obs = []
    while len(obs) < 512:
        z = rng.gauss(0, 1); d = 24.6 + 1.95 * z + 0.22 * (z * z - 1)
        if 19.5 <= d <= 31.5:
            c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0, 0.21)
            obs.append((round(d, 2), round(min(0.97, max(0.02, c)), 3)))
    sim = []
    while len(sim) < 1000:
        b = rng.uniform(-lim, lim); d = 24.5 + 1.42 * b + 0.10 * (b * b - 1.33)
        if 19.5 <= d <= 31.5:
            c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0, 0.15)
            sim.append((round(d, 2), round(min(cap, max(0.02, c)), 3)))
    return obs, sim


def ps(size, d, c, t=0.13, klo=1200.0, khi=25.0, kc=4.0, b0=4.6):
    o = (size - d) / d
    if o < 0 or o > 0.45: return 0.0
    pen = klo * (t - o) ** 2 if o < t else khi * (o - t) ** 2
    return 1 / (1 + math.exp(-(b0 - pen - kc * c)))


def psucc(d, c): return max(ps(s, d, c) for s in SIZES)
def meanp(co): return sum(psucc(d, c) for d, c in co) / len(co)


obs, sim = cohorts()
do = [d for d, _ in obs]; ds = [d for d, _ in sim]
co = [c for _, c in obs]; cs = [c for _, c in sim]

print("== 1. Cohort coverage ==")
for n, g in (("observed", obs), ("simulated", sim)):
    dd = [d for d, _ in g]; cc = [c for _, c in g]
    print(f"{n:<10} n={len(g):>4}  diameter mean={statistics.mean(dd):6.3f} "
          f"sd={statistics.stdev(dd):5.3f}  calc mean={statistics.mean(cc):5.3f} "
          f"sd={statistics.stdev(cc):5.3f}")
print(f"SMD diameter                : {smd(statistics.mean(ds), statistics.stdev(ds), statistics.mean(do), statistics.stdev(do)):+.3f}")
print(f"SD ratio diameter           : {statistics.stdev(ds) / statistics.stdev(do):.3f}")
print(f"SMD calcification           : {smd(statistics.mean(cs), statistics.stdev(cs), statistics.mean(co), statistics.stdev(co)):+.3f}")
print(f"SD ratio calcification      : {statistics.stdev(cs) / statistics.stdev(co):.3f}")
crit = 1.36 * math.sqrt((len(obs) + len(sim)) / (len(obs) * len(sim)))
print(f"KS diameter                 : {ks(do, ds):.4f}   (5% critical value {crit:.4f})")
print(f"KS calcification            : {ks(co, cs):.4f}   (5% critical value {crit:.4f})")
for lab, o, s in (("diameter", do, ds), ("calcification", co, cs)):
    print(f"{lab:<14} observed P1-P99 {q(o, 0.01):6.3f}-{q(o, 0.99):6.3f}  "
          f"simulated P1-P99 {q(s, 0.01):6.3f}-{q(s, 0.99):6.3f}  "
          f"coverage {sum(1 for x in o if q(s, 0.01) <= x <= q(s, 0.99)) / len(o):.3f}")
print(f"observed calcification > 0.72 : {sum(1 for c in co if c > 0.72) / len(co):.3f}")

print("\n== 2. Predicted success ==")
a, b = meanp(sim), meanp(obs)
print(f"simulated {a:.4f}   observed {b:.4f}   optimism {a - b:+.4f} ({100 * (a - b):+.2f} pp)")
geom = sum(psucc(d, 0.30) for d in ds) / len(ds) - sum(psucc(d, 0.30) for d in do) / len(do)
calc = sum(psucc(24.5, c) for c in cs) / len(cs) - sum(psucc(24.5, c) for c in co) / len(co)
print(f"geometry component        : {geom:+.4f}")
print(f"calcification component   : {calc:+.4f}")
print(f"interaction               : {a - b - geom - calc:+.4f}")
ys = sorted(obs, key=lambda t: t[0]); st = len(ys) // 10
for i in range(10):
    ch = ys[i * st:(i + 1) * st] if i < 9 else ys[9 * st:]
    print(f"decile {i + 1:>2}  D {ch[0][0]:5.2f}-{ch[-1][0]:5.2f}  "
          f"C {statistics.mean(c for _, c in ch):.3f}  success {meanp(ch):.3f}")
print(f"Wilson simulated : {wilson(int(a * 1000), 1000)[0]:.4f} to {wilson(int(a * 1000), 1000)[1]:.4f}")
print(f"Wilson observed  : {wilson(int(b * 512), 512)[0]:.4f} to {wilson(int(b * 512), 512)[1]:.4f}")

print("\n== 3. Sensitivity to truncation ==")
for lim in (1.5, 2.0, 2.5, 3.0):
    _, s2 = cohorts(lim=lim)
    p = meanp(s2)
    print(f"+/-{lim}  sim sd {statistics.stdev(d for d, _ in s2):.3f}  "
          f"p_sim {p:.4f}  p_obs {b:.4f}  optimism {p - b:+.4f}")

print("\n== 4. Restricted to 21.0-27.0 mm ==")
o2 = [t for t in obs if 21.0 <= t[0] <= 27.0]; s2 = [t for t in sim if 21.0 <= t[0] <= 27.0]
print(f"retained {len(o2)}/{len(obs)} observed, {len(s2)}/{len(sim)} simulated")
print(f"observed {meanp(o2):.4f}  simulated {meanp(s2):.4f}  optimism {meanp(s2) - meanp(o2):+.4f}")
print(f"observed outside range: {1 - len(o2) / len(obs):.3f}")

print("\n== 5. Calibration and validation ==")
rng = random.Random(SEED + 7); sd = 0.34; rows = []
for _ in range(48):
    d = rng.uniform(4, 12); rows.append((d, 0.62 * d + 0.9 + rng.gauss(0, sd)))
cal, val = rows[0::2], rows[1::2]


def poly(dat, deg):
    m = deg + 1
    A = [[sum(x ** (i + j) for x, _ in dat) for j in range(m)] for i in range(m)]
    v = [sum(x ** i * y for x, y in dat) for i in range(m)]
    for i in range(m):
        pv = A[i][i]; A[i] = [z / pv for z in A[i]]; v[i] /= pv
        for k in range(m):
            if k != i:
                f = A[k][i]; A[k] = [A[k][j] - f * A[i][j] for j in range(m)]; v[k] -= f * v[i]
    return v


def rmse(dat, cf): return math.sqrt(sum((y - sum(c * x ** i for i, c in enumerate(cf))) ** 2 for x, y in dat) / len(dat))


for name, deg in (("linear", 1), ("quadratic", 2), ("cubic", 3), ("quartic", 4)):
    cf = poly(cal, deg)
    print(f"{name:>10}  calib {rmse(cal, cf):.4f}  valid {rmse(val, cf):.4f}")
q4 = poly(cal, 4)
print(f"repeatability 0.34 N; reported 'validation' {rmse(cal, q4):.4f} N; honest validation {rmse(val, q4):.4f} N")
print(f"dof factor sqrt((24-5)/24) = {math.sqrt(19 / 24):.3f}; expected fit RMSE {0.34 * math.sqrt(19 / 24):.4f} N")

"""Cohort coverage, bias decomposition and calibration audit for an in-silico trial.

Standard library only. Deterministic: every random draw is seeded.
Run:  python cohort_audit.py
"""

import math
import random
import statistics

SEED = 20261027
SIZES = (20.0, 22.0, 24.0, 26.0, 28.0, 30.0)   # commercial device sizes, mm


# ------------------------------------------------------------------ helpers
def quantile(xs, p):
    ys = sorted(xs)
    if len(ys) == 1:
        return ys[0]
    pos = p * (len(ys) - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] + (pos - lo) * (ys[hi] - ys[lo])


def ks_two_sample(a, b):
    """Two-sample Kolmogorov-Smirnov statistic on the empirical CDFs."""
    xs = sorted(set(a) | set(b))
    sa, sb = sorted(a), sorted(b)
    ia = ib = 0
    best = 0.0
    for x in xs:
        while ia < len(sa) and sa[ia] <= x:
            ia += 1
        while ib < len(sb) and sb[ib] <= x:
            ib += 1
        best = max(best, abs(ia / len(sa) - ib / len(sb)))
    return best


def smd(mean_a, sd_a, mean_b, sd_b):
    pooled = math.sqrt((sd_a ** 2 + sd_b ** 2) / 2.0)
    return (mean_a - mean_b) / pooled


def wilson(k, n, z=1.959964):
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half


# -------------------------------------------------------------- cohorts
def build_cohorts(lim=2.0, cap=0.72):
    """Observed registry cohort and SSM-generated virtual cohort.

    Each patient is (annulus diameter mm, calcification index 0-1).
    """
    rng = random.Random(SEED)
    observed = []
    while len(observed) < 512:
        z = rng.gauss(0.0, 1.0)
        d = 24.6 + 1.95 * z + 0.22 * (z * z - 1.0)
        if not (19.5 <= d <= 31.5):
            continue
        c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0.0, 0.21)
        observed.append((round(d, 2), round(min(0.97, max(0.02, c)), 3)))

    simulated = []
    while len(simulated) < 1000:
        b = rng.uniform(-lim, lim)          # shape-mode amplitude, SD units
        d = 24.5 + 1.42 * b + 0.10 * (b * b - 1.33)
        if not (19.5 <= d <= 31.5):
            continue
        c = 0.30 + 0.045 * (d - 24.0) + rng.gauss(0.0, 0.15)
        simulated.append((round(d, 2), round(min(cap, max(0.02, c)), 3)))
    return observed, simulated


# --------------------------------------------------- device and outcome model
def p_single(size, d, c, target=0.13, klo=1200.0, khi=25.0, kcalc=4.0, b0=4.6):
    """Predicted probability of procedural success for one implanted size."""
    o = (size - d) / d
    if o < 0.0 or o > 0.45:
        return 0.0
    pen = klo * (target - o) ** 2 if o < target else khi * (o - target) ** 2
    return 1.0 / (1.0 + math.exp(-(b0 - pen - kcalc * c)))


def p_success(d, c):
    """The clinician picks the size with the best predicted outcome."""
    return max(p_single(s, d, c) for s in SIZES)


def mean_p(cohort):
    return sum(p_success(d, c) for d, c in cohort) / len(cohort)


# ------------------------------------------------------------------ blocks
def cohort_block(obs, sim):
    print("== 1. Cohort coverage ==")
    for name, coh in (("observed", obs), ("simulated", sim)):
        ds = [d for d, _ in coh]
        cs = [c for _, c in coh]
        print(f"{name:<10} n={len(coh):>4}  diameter mean={statistics.mean(ds):6.3f} "
              f"sd={statistics.stdev(ds):5.3f}  calc mean={statistics.mean(cs):5.3f} "
              f"sd={statistics.stdev(cs):5.3f}")

    do = [d for d, _ in obs]
    ds_ = [d for d, _ in sim]
    co = [c for _, c in obs]
    cs_ = [c for _, c in sim]

    print(f"SMD diameter                : "
          f"{smd(statistics.mean(ds_), statistics.stdev(ds_), statistics.mean(do), statistics.stdev(do)):+.3f}")
    print(f"SD ratio diameter           : {statistics.stdev(ds_) / statistics.stdev(do):.3f}")
    print(f"SMD calcification           : "
          f"{smd(statistics.mean(cs_), statistics.stdev(cs_), statistics.mean(co), statistics.stdev(co)):+.3f}")
    print(f"SD ratio calcification      : {statistics.stdev(cs_) / statistics.stdev(co):.3f}")

    ks_d = ks_two_sample(do, ds_)
    ks_c = ks_two_sample(co, cs_)
    crit = 1.36 * math.sqrt((len(obs) + len(sim)) / (len(obs) * len(sim)))
    print(f"KS diameter                 : {ks_d:.4f}   (5% critical value {crit:.4f})")
    print(f"KS calcification            : {ks_c:.4f}   (5% critical value {crit:.4f})")

    for label, o, s in (("diameter", do, ds_), ("calcification", co, cs_)):
        lo_o, hi_o = quantile(o, 0.01), quantile(o, 0.99)
        lo_s, hi_s = quantile(s, 0.01), quantile(s, 0.99)
        inside = sum(1 for x in o if lo_s <= x <= hi_s) / len(o)
        print(f"{label:<14} observed P1-P99 {lo_o:6.3f}-{hi_o:6.3f}  "
              f"simulated P1-P99 {lo_s:6.3f}-{hi_s:6.3f}  coverage {inside:.3f}")
    print(f"observed with calcification > 0.72: {sum(1 for c in co if c > 0.72) / len(co):.3f}")
    print(f"simulated with calcification > 0.72: {sum(1 for c in cs_ if c > 0.72) / len(cs_):.3f}")


def bias_block(obs, sim):
    print("\n== 2. Predicted success: simulated versus observed ==")
    ps, po = mean_p(sim), mean_p(obs)
    print(f"mean predicted success, simulated : {ps:.4f}")
    print(f"mean predicted success, observed  : {po:.4f}")
    print(f"optimism                          : {ps - po:+.4f}  ({100 * (ps - po):+.2f} pp)")

    d_obs = [d for d, _ in obs]
    c_obs = [c for _, c in obs]
    d_sim = [d for d, _ in sim]
    c_sim = [c for _, c in sim]
    geom = sum(p_success(d, 0.30) for d in d_sim) / len(d_sim) \
        - sum(p_success(d, 0.30) for d in d_obs) / len(d_obs)
    calc = sum(p_success(24.5, c) for c in c_sim) / len(c_sim) \
        - sum(p_success(24.5, c) for c in c_obs) / len(c_obs)
    print("\n-- one-at-a-time decomposition of the optimism --")
    print(f"geometry only, calcification fixed at 0.30 : {geom:+.4f}")
    print(f"calcification only, diameter fixed at 24.5 : {calc:+.4f}")
    print(f"sum of parts                               : {geom + calc:+.4f}")
    print(f"actual joint difference                    : {ps - po:+.4f}")

    print("\n-- predicted success by observed diameter decile --")
    ys = sorted(obs, key=lambda t: t[0])
    step = len(ys) // 10
    for i in range(10):
        chunk = ys[i * step:(i + 1) * step] if i < 9 else ys[9 * step:]
        print(f"decile {i + 1:>2}  n={len(chunk):>3}  D {chunk[0][0]:5.2f}-{chunk[-1][0]:5.2f}  "
              f"mean C {statistics.mean(c for _, c in chunk):.3f}  "
              f"predicted success {mean_p(chunk):.3f}")

    n = len(sim)
    lo, hi = wilson(int(round(ps * n)), n)
    lo2, hi2 = wilson(int(round(po * len(obs))), len(obs))
    print(f"\nWilson 95% CI, simulated {ps:.4f} (n={n})   : {lo:.4f} to {hi:.4f}")
    print(f"Wilson 95% CI, observed  {po:.4f} (n={len(obs)}) : {lo2:.4f} to {hi2:.4f}")
    print(f"claim threshold 0.93 -> simulated {'PASS' if lo >= 0.93 else 'FAIL'}, "
          f"observed {'PASS' if lo2 >= 0.93 else 'FAIL'}")


def sensitivity_block(obs):
    print("\n== 3. Sensitivity: shape-mode truncation and calcification cap ==")
    print(f"{'trunc':>7} {'C cap':>6} {'sim D sd':>9} {'sim C p99':>10} "
          f"{'p_sim':>7} {'p_obs':>7} {'optimism':>9}")
    for lim in (1.5, 2.0, 2.5, 3.0):
        for cap in (0.72, 0.97):
            _, sim = build_cohorts(lim=lim, cap=cap)
            ps, po = mean_p(sim), mean_p(obs)
            print(f"{'+/-' + str(lim):>7} {cap:6.2f} "
                  f"{statistics.stdev(d for d, _ in sim):9.3f} "
                  f"{quantile([c for _, c in sim], 0.99):10.3f} "
                  f"{ps:7.4f} {po:7.4f} {ps - po:+9.4f}")


def restricted_block(obs, sim):
    print("\n== 4. Claim restricted to the labelled range 21.0-27.0 mm ==")
    o = [t for t in obs if 21.0 <= t[0] <= 27.0]
    s = [t for t in sim if 21.0 <= t[0] <= 27.0]
    print(f"observed retained {len(o)}/{len(obs)}, simulated retained {len(s)}/{len(sim)}")
    print(f"mean predicted success, observed in range  : {mean_p(o):.4f}")
    print(f"mean predicted success, simulated in range : {mean_p(s):.4f}")
    print(f"optimism in range                          : {mean_p(s) - mean_p(o):+.4f}")
    print(f"observed patients outside 21.0-27.0 mm     : {1 - len(o) / len(obs):.3f}")


def calibration_block():
    print("\n== 5. Calibration and validation separation ==")
    rng = random.Random(SEED + 7)
    repeat_sd = 0.34
    rows = []
    for _ in range(48):
        d = rng.uniform(4.0, 12.0)
        y = 0.62 * d + 0.9 + rng.gauss(0.0, repeat_sd)
        rows.append((d, y))
    calib, valid = rows[0::2], rows[1::2]   # stratified by compression: alternating after sort
    print(f"specimens: {len(calib)} calibration, {len(valid)} validation; "
          f"bench repeatability {repeat_sd:.2f} N (3 repeats per specimen)")

    def polyfit(data, degree):
        m = degree + 1
        A = [[sum(x ** (i + j) for x, _ in data) for j in range(m)] for i in range(m)]
        b = [sum((x ** i) * y for x, y in data) for i in range(m)]
        for i in range(m):
            piv = A[i][i]
            A[i] = [v / piv for v in A[i]]
            b[i] /= piv
            for k in range(m):
                if k != i:
                    f = A[k][i]
                    A[k] = [A[k][j] - f * A[i][j] for j in range(m)]
                    b[k] -= f * b[i]
        return b

    def pred(coef, x):
        return sum(c * x ** i for i, c in enumerate(coef))

    def rmse(data, coef):
        return math.sqrt(sum((y - pred(coef, x)) ** 2 for x, y in data) / len(data))

    print("\n-- candidate model forms, fitted on the 24 calibration specimens --")
    print(f"{'form':>12} {'calib RMSE':>11} {'valid RMSE':>11}")
    coefs = {}
    for name, deg in (("linear", 1), ("quadratic", 2), ("cubic", 3), ("quartic", 4)):
        c = polyfit(calib, deg)
        coefs[name] = c
        print(f"{name:>12} {rmse(calib, c):11.4f} {rmse(valid, c):11.4f}")

    print(f"\nreported RMSE if the calibration fit is presented as validation: "
          f"{rmse(calib, coefs['quartic']):.4f} N")
    print(f"honest validation RMSE, same form                                : "
          f"{rmse(valid, coefs['quartic']):.4f} N")
    print(f"bench repeatability                                              : {repeat_sd:.2f} N")
    print(f"calibration RMSE below repeatability?                            : "
          f"{'YES - the fit has absorbed the noise' if rmse(calib, coefs['quartic']) < repeat_sd else 'no'}")

    n, p_par = len(calib), 5
    factor = math.sqrt((n - p_par) / n)
    print(f"\ndegrees-of-freedom factor sqrt((n-p)/n)  : {factor:.3f}  (n={n}, p={p_par})")
    print(f"expected calibration RMSE for a correct  : {repeat_sd * factor:.4f} N")
    print(f"model fitted to noise alone              ")
    print(f"linear validation RMSE                   : {rmse(valid, coefs['linear']):.4f} N")


def precision_block():
    print("\n== 6. Precision of a simulated success rate ==")
    for n in (100, 200, 500, 1000, 2000, 5000):
        lo, hi = wilson(int(round(0.946 * n)), n)
        print(f"n={n:>5}  point estimate 0.946  half-width {(hi - lo) / 2 * 100:5.2f} pp")


def main():
    obs, sim = build_cohorts()
    cohort_block(obs, sim)
    bias_block(obs, sim)
    sensitivity_block(obs)
    restricted_block(obs, sim)
    calibration_block()
    precision_block()


if __name__ == "__main__":
    main()

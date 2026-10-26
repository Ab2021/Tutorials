"""CUSUM monitoring for a performance metric.

The CUSUM accumulates deviations from a target. k is the reference value
(half the shift to detect); h is the decision interval. Both are in the
units of the standardised deviation.
"""

import math

# Monthly AUROC after deployment, with the number of labelled cases per month.
monthly = [
    (1, 0.821, 640), (2, 0.819, 655), (3, 0.815, 702), (4, 0.808, 688),
    (5, 0.799, 731), (6, 0.785, 745), (7, 0.772, 719), (8, 0.761, 760),
    (9, 0.748, 742), (10, 0.741, 755), (11, 0.738, 748), (12, 0.735, 770),
]

TARGET = 0.820      # validated performance
SHIFT = 0.030       # smallest drop we care to detect quickly
K = SHIFT / 2       # reference value
H = 5 * 0.012       # decision interval, in AUROC units (five sigma)


def cusum(values, target, k, h):
    s_hi = 0.0
    out = []
    for i, (month, x, n) in enumerate(values, start=1):
        s_hi = max(0.0, s_hi + (target - x) - k)
        out.append((month, x, s_hi, s_hi > h))
    return out


print(f"target {TARGET:.3f}, shift {SHIFT:.3f}, k {K:.3f}, h {H:.3f}")
print(f"{'month':>6}{'auroc':>8}{'cusum':>8}  alarm")
print("-" * 32)
alarm_month = None
for month, x, s, alarm in cusum(monthly, TARGET, K, H):
    print(f"{month:>6}{x:>8.3f}{s:>8.3f}  {'YES' if alarm else ''}")
    if alarm and alarm_month is None:
        alarm_month = month

print()
print("first alarm month:", alarm_month)

# Expected detection delay for a sustained shift of SHIFT, in months.
# For a one-sided CUSUM, the average run length after a shift is roughly h / (shift - k).
arl = H / (SHIFT - K)
print(f"approximate ARL after shift: {arl:.1f} months")

# Sample size for detecting a drop of 0.05 in AUROC at 80% power (normal approximation).
# SE(AUROC) ~ 0.63 * (1 - 0.63) / sqrt(n_pos) is not exact; use the Hanley-McNeil form.
def se_auroc(a, n_pos, n_neg):
    q1 = a / (2 - a)
    q2 = 2 * a * a / (1 + a)
    return math.sqrt((a * (1 - a) + (n_pos - 1) * (q1 - a * a)
                      + (n_neg - 1) * (q2 - a * a)) / (n_pos * n_neg))


a = 0.82
n_pos, n_neg = 150, 850          # 15% prevalence, 1,000 cases
se = se_auroc(a, n_pos, n_neg)
z_alpha, z_beta = 1.96, 0.84
print()
print(f"SE(AUROC) at n=1000, 15% prevalence: {se:.4f}")
for delta in (0.05, 0.02):
    multiplier = ((z_alpha + z_beta) * se / delta) ** 2
    print(f"  detect a {delta:.2f} drop: {multiplier:.2f}x the sample, "
          f"about {multiplier*1000:.0f} labelled cases")

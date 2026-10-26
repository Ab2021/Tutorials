strata = [
    ("young", 2000, 0.20, 0.5),
    ("middle", 2000, 0.35, 1.0),
    ("older", 1000, 0.50, 2.5),
]
n = sum(row[1] for row in strata)
unweighted_num = sum(row[1] * row[2] for row in strata)
weighted_num = sum(row[1] * row[2] * row[3] for row in strata)
weight_sum = sum(row[1] * row[3] for row in strata)

print(f"{'Stratum':<10}{'n':>7}{'Prevalence':>12}{'Weight':>8}{'Weighted n':>12}")
for name, count, prev, w in strata:
    print(f"{name:<10}{count:>7}{prev:>12.0%}{w:>8.1f}{count*w:>12,.0f}")
print()
print(f"Unweighted prevalence: {unweighted_num/n:.3f}")
print(f"Weighted prevalence:   {weighted_num/weight_sum:.3f}")
print(f"Difference:            {weighted_num/weight_sum - unweighted_num/n:+.3f}")

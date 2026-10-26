tests = 20000
alpha = 0.05
bonf = alpha / tests
expected_fp = tests * alpha

print(f"Tests: {tests:,}")
print(f"Nominal alpha: {alpha}")
print(f"Bonferroni threshold: {bonf:.2e}")
print(f"Expected false positives at nominal alpha: {expected_fp:,.0f}")
print()
p_values = [0.000001, 0.00001, 0.0004, 0.002, 0.01, 0.03, 0.2, 0.6]
print(f"{'p-value':>12}{'Bonferroni significant':>24}")
for p in p_values:
    print(f"{p:>12.6f}{str(p < bonf):>24}")
print()
print("With 20,000 genes, a nominal 0.05 threshold yields about 1,000 false positives")
print("before any true signal is counted. Multiple-testing control is not optional.")

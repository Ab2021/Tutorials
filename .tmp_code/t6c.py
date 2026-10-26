from itertools import combinations

group_a = [8.1, 7.4, 8.8]   # donor-level pseudobulk means
group_b = [6.9, 7.2, 6.4]
observed = sum(group_a) / len(group_a) - sum(group_b) / len(group_b)
pooled = group_a + group_b
n = len(group_a)
extreme = 0
total = 0
for idx in combinations(range(len(pooled)), n):
    a = [pooled[i] for i in idx]
    b = [pooled[i] for i in range(len(pooled)) if i not in idx]
    diff = sum(a) / len(a) - sum(b) / len(b)
    total += 1
    if abs(diff) >= abs(observed) - 1e-12:
        extreme += 1

print(f"Observed difference: {observed:.2f}")
print(f"Permutations: {total}")
print(f"Two-sided permutation p-value: {extreme/total:.3f}")
print()
print("With three donors per group, the smallest attainable two-sided p-value is 0.10.")
print("No donor-level claim can reach conventional significance at this sample size.")

def dice(pred, truth):
    inter = len(pred & truth)
    return 2 * inter / (len(pred) + len(truth))

cases = [
    ("large lesion, good overlap", set(range(1000)), set(range(50, 1050))),
    ("small lesion, shifted", set(range(100)), set(range(60, 160))),
    ("small lesion, missed edge", set(range(100)), set(range(90, 190))),
]
print(f"{'Case':<32}{'Pred':>6}{'Truth':>7}{'Dice':>8}")
for name, pred, truth in cases:
    print(f"{name:<32}{len(pred):>6}{len(truth):>7}{dice(pred, truth):>8.3f}")
print()
print("Dice is insensitive to absolute size: a small missed lesion and a large")
print("partially covered lesion can produce similar values.")

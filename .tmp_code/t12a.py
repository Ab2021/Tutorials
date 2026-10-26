N = 1000
for contaminated_share in (0.10, 0.20, 0.40):
    c = int(N * contaminated_share)
    clean = N - c
    acc_c, acc_clean = 0.95, 0.70
    overall = (c * acc_c + clean * acc_clean) / N
    inflation = overall - acc_clean
    print(f"Contaminated {contaminated_share:.0%}: {c} items, reported {overall:.1%}, clean {acc_clean:.1%}, inflation {inflation:+.1%}")
print()
print("The clean estimate is the number that generalizes; the reported number includes memorisation.")

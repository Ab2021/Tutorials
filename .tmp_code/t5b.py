def ppv(sensitivity, specificity, prevalence):
    tp = sensitivity * prevalence
    fp = (1 - specificity) * (1 - prevalence)
    return tp / (tp + fp)

print(f"{'Prevalence':>12}{'PPV':>10}{'False alarms per 1,000':>24}")
for prev in (0.001, 0.005, 0.01, 0.05, 0.10, 0.20):
    value = ppv(0.95, 0.97, prev)
    fp_per_1000 = round((1 - 0.97) * (1 - prev) * 1000)
    print(f"{prev:>12.1%}{value:>10.1%}{fp_per_1000:>24}")
print()
print("Assumptions: sensitivity 0.95, specificity 0.97.")
print("At 0.5% prevalence, about 14% of alerts are true.")

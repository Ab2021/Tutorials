# Litigation & Large Loss Modelling (Part 1) - Theoretical Deep Dive

## Overview
One \$10M claim can distort the entire triangle for a small insurer. Standard Chain Ladder fails when "Large Losses" are present because they develop differently from "Attritional Losses." We explore the **Bifurcation Method** (splitting the triangle), **Excess Loss Factors (ELF)**, and the statistical distributions (**Pareto, LogNormal**) used to model the tail.

---

## 1. Conceptual Foundation

### 1.1 The Distortion Problem

**Scenario:**
*   Year 2020 has \$1M in paid losses at Age 12.
*   Year 2021 has \$5M (due to one \$4M shock loss).
*   **Chain Ladder:** If LDF=2.0, Year 2021 projects to \$10M.
*   **Reality:** The \$4M claim might be settled. It won't double. The \$1M attritional part will double.
*   **Conclusion:** Applying the same LDF to Large and Small losses is a fundamental error.

### 1.2 Defining "Large"

*   **Threshold:** A fixed dollar amount (e.g., \$100k) or a percentile (99th).
*   **Attritional:** Claims < Threshold. (High frequency, low severity).
*   **Large:** Claims > Threshold. (Low frequency, high severity).
*   **Catastrophe:** A single event causing multiple large claims (e.g., Hurricane). Modeled separately.

### 1.3 Excess Loss Factors (ELF)

*   **ELF:** The ratio of Excess Losses to Total Losses (or Premium).
*   Used to "load" the large losses back in after modeling the attritional part.
*   **Formula:** $ELF(L) = \frac{E[\text{Losses} > L]}{E[\text{Total Losses}]}$.

---

## 2. Mathematical Framework

### 2.1 The Bifurcation Method

1.  **Split the Data:**
    *   $C_{attr}$: Triangle of losses capped at \$100k.
    *   $C_{large}$: Triangle of the excess portion (or counts of large claims).
2.  **Develop Attritional:**
    *   Run Chain Ladder on $C_{attr}$. Get $U_{attr}$.
3.  **Load Large:**
    *   **Method A (Development):** Run Chain Ladder on $C_{large}$ (if enough data).
    *   **Method B (Loading):** $U_{large} = U_{attr} \times \text{Load Factor}$.
    *   **Method C (Frequency/Severity):** $U_{large} = E[\text{Count}] \times E[\text{Severity}]$.

### 2.2 Modeling Severity Distributions

**LogNormal:**
*   Good for the "body" of the distribution.
*   $f(x) = \frac{1}{x\sigma\sqrt{2\pi}} \exp\left(-\frac{(\ln x - \mu)^2}{2\sigma^2}\right)$.

**Pareto (Heavy Tail):**
*   Good for the "tail" (Large Losses).
*   $F(x) = 1 - (\frac{\theta}{x+\theta})^\alpha$.
*   **Key Property:** The mean excess loss increases linearly with the threshold.

### 2.3 Splicing

*   Use LogNormal for claims < \$100k.
*   Use Pareto for claims > \$100k.
*   Ensure the PDF is continuous at the splice point.

---

## 3. Theoretical Properties

### 3.1 The "Survival" of Large Losses

*   Attritional claims close fast.
*   Large claims stay open (Litigation).
*   **Implication:** The LDF for Large Losses is usually *higher* and *longer* than for Attritional.
*   *Exception:* If "Large" means "Total Loss" (e.g., car totaled), it might close fast. Context matters.

### 3.2 Parameter Uncertainty

*   Estimating the Pareto $\alpha$ (shape) is notoriously difficult with small samples.
*   **Hill Estimator:** A standard method for $\alpha$, but volatile.
*   **Impact:** A small change in $\alpha$ (e.g., 1.5 to 1.4) can double the tail reserve.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting a Composite Model (Python)

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import lognorm, pareto

# Data: Individual Claim Severities
# 90% small, 10% large
np.random.seed(42)
n = 1000
small_claims = np.random.lognormal(mean=8, sigma=1, size=900) # Mean ~3000
large_claims = (np.random.pareto(a=2, size=100) + 1) * 50000 # Mean ~100k

claims = np.concatenate([small_claims, large_claims])
threshold = 50000

# 1. Split Data
body = claims[claims < threshold]
tail = claims[claims >= threshold]

# 2. Fit LogNormal to Body
shape, loc, scale = lognorm.fit(body, floc=0)
print(f"LogNormal Sigma: {shape:.3f}, Scale: {scale:.1f}")

# 3. Fit Pareto to Tail
# Pareto Type II (Lomax) is often better, but using simplified Pareto here
# x_m = threshold
# alpha = n / sum(ln(x / x_m))
alpha = len(tail) / np.sum(np.log(tail / threshold))
print(f"Pareto Alpha: {alpha:.3f}")

# 4. Calculate Expected Severity
# E[Body] (Truncated LogNormal) + E[Tail] * P(Tail)
# Simplified: Just use the fitted means
mean_body = np.mean(body) # Or analytical
mean_tail = (alpha * threshold) / (alpha - 1) # Analytical Pareto Mean

total_mean = mean_body * (len(body)/n) + mean_tail * (len(tail)/n)
print(f"Modeled Mean Severity: {total_mean:.2f}")
print(f"Actual Mean Severity: {np.mean(claims):.2f}")

# Visualization
plt.hist(claims, bins=50, log=True, alpha=0.6, label='Data')
plt.axvline(threshold, color='r', linestyle='--', label='Splice Point')
plt.title("Composite Loss Distribution")
plt.legend()
plt.show()
```

### 4.2 Calculating Excess Loss Factors (ELF)

```python
# ELF at Threshold L
# ELF = E[X - L | X > L] * P(X > L) / E[X]
#     = (Mean Excess Loss * Probability) / Total Mean

def calculate_elf(claims, limit):
    # Capped Claims
    capped = np.minimum(claims, limit)
    # Excess portion
    excess = np.maximum(claims - limit, 0)
    
    total_loss = np.sum(claims)
    total_excess = np.sum(excess)
    
    return total_excess / total_loss

limits = [10000, 50000, 100000, 500000]
elfs = [calculate_elf(claims, l) for l in limits]

print("\nExcess Loss Factors:")
for l, e in zip(limits, elfs):
    print(f"Limit {l}: {e:.3f}")

# Interpretation:
# If ELF at 50k is 0.40, it means 40% of the total dollars come from the layer above 50k.
# We can reserve the first 60% using Chain Ladder, and load the 40% using an ELF.
```

---

## 5. Evaluation & Validation

### 5.1 The "Mean Excess Plot"

*   Plot $E[X - u | X > u]$ against threshold $u$.
*   **Pareto Property:** If the tail is Pareto, this plot should be a straight line with positive slope.
*   If it curves down, the tail is lighter (LogNormal). If it curves up, the tail is super-heavy.

### 5.2 Stability Check

*   Calculate the ELF for 2018, 2019, 2020.
*   Is it stable?
*   **Social Inflation:** If ELF is rising (0.30 $\to$ 0.35 $\to$ 0.40), large losses are growing faster than attritional ones. You must trend the ELF.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: ALAE Inclusion**
    *   **Issue:** Do you include Allocated Loss Adjustment Expense (Defense Costs) in the definition of "Large"?
    *   **Reality:** Yes. A \$10k claim with \$200k in legal fees is a Large Loss.
    *   **Fix:** Use "Loss + ALAE" for the threshold.

2.  **Trap: Inflation on the Threshold**
    *   **Issue:** A \$100k claim in 1990 is a \$300k claim today.
    *   **Reality:** If you use a fixed \$100k threshold, more and more attritional claims will "creep" into the large bucket over time.
    *   **Fix:** Index the threshold (e.g., \$100k in 2010, \$120k in 2015).

### 6.2 Implementation Challenges

1.  **Sparse Data:**
    *   You might have only 5 large claims in history.
    *   **Solution:** You cannot model this. Use Industry ELFs (ISO/NCCI) or Bornhuetter-Ferguson with a Large Loss ELR.

---

## 7. Advanced Topics & Extensions

### 7.1 Extreme Value Theory (EVT)

*   **GPD (Generalized Pareto Distribution):** The theoretical limit distribution for excesses over a high threshold.
*   **Peaks Over Threshold (POT):** A method to estimate GPD parameters efficiently.

### 7.2 Catastrophe Modeling

*   For Hurricanes/Earthquakes, we don't use triangles.
*   We use **RMS/AIR** models (Physics-based simulations).
*   The "Large Loss Load" for Property is often replaced by the "CAT Load" from these models.

---

## 8. Regulatory & Governance Considerations

### 8.1 Reinsurance Recoverables

*   Large losses are usually reinsured.
*   **Gross vs. Net:**
    *   Gross Reserve = Attritional + Large.
    *   Net Reserve = Attritional + (Large - Reinsurance Recovery).
*   **Credit Risk:** If the reinsurer goes bust, the Net Reserve jumps back to Gross.

---

## 9. Practical Example

### 9.1 Worked Example: The "Capped" Triangle

**Scenario:**
*   Total Paid to Date: \$50M.
*   Large Claims (> \$500k): \$10M (2 claims).
*   Attritional Paid: \$40M.

**Method:**
1.  **Attritional:** Develop \$40M with LDF 1.5 $\to$ \$60M Ultimate.
2.  **Large:**
    *   Don't develop the \$10M (too volatile).
    *   Use an ELF. Industry ELF at \$500k is 0.20.
    *   Implies Large = 20% of Total, Attritional = 80%.
    *   Total Ultimate = \$60M / 0.80 = \$75M.
    *   Implies Large Ultimate = \$15M.
3.  **Reserve:** \$75M - \$50M = \$25M.

**Comparison:**
*   If we developed the full \$50M with LDF 1.5 $\to$ \$75M.
*   *Coincidence?* In this case, yes. But if the Large claims were \$20M, standard CL would project \$90M, while the ELF method would constrain it based on the attritional signal.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Bifurcate:** Always look at Large and Small separately.
2.  **Pareto:** The king of tail distributions.
3.  **ELF:** The bridge between the two worlds.

### 10.2 When to Use This Knowledge
*   **Commercial Liability:** Where large losses dominate.
*   **Reinsurance Pricing:** Estimating the cost of the layer \$1M xs \$1M.

### 10.3 Critical Success Factors
1.  **Define the Threshold Carefully:** Too low = too much noise. Too high = no data.
2.  **Watch for Trends:** Is the ELF increasing? (Social Inflation).
3.  **Don't ignore ALAE:** Lawyers are expensive.

### 10.4 Further Reading
*   **Miccolis:** "On the Theory of Increased Limits and Excess of Loss Pricing".
*   **Reinsurance Association of America:** "Historical Loss Development Study" (Source of ELFs).

---

## Appendix

### A. Glossary
*   **ELF:** Excess Loss Factor.
*   **Lomax:** Pareto Type II (shifted Pareto).
*   **Splicing:** Joining two distributions.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Pareto PDF** | $\alpha \theta^\alpha / (x+\theta)^{\alpha+1}$ | Tail Modeling |
| **ELF** | $E[Excess] / E[Total]$ | Loading |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

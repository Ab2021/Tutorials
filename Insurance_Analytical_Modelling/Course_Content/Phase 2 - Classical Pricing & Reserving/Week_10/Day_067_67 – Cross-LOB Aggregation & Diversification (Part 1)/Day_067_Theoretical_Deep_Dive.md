# Cross-LOB Aggregation & Diversification (Part 1) - Theoretical Deep Dive

## Overview
Diversification is the "Only Free Lunch" in finance. By combining risks that don't crash together, we reduce the total capital required. This session covers **Correlation Matrices**, **Diversification Benefit**, and the difference between **Pearson** and **Rank** correlation.

---

## 1. Conceptual Foundation

### 1.1 The Diversification Benefit

*   **Formula:** $Div = \sum SCR_i - SCR_{total}$.
*   **Source:**
    *   **Inter-Risk:** Market Risk vs. Insurance Risk. (Usually low correlation).
    *   **Intra-Risk:** Motor vs. Property. (Medium correlation).
    *   **Geographic:** US Hurricane vs. Japan Earthquake. (Zero correlation).

### 1.2 Correlation Measures

*   **Pearson Correlation:** Measures *linear* dependence.
    *   *Flaw:* Sensitive to outliers. Assumes Elliptical distribution (Normal).
*   **Rank Correlation (Spearman/Kendall):** Measures *monotonic* dependence.
    *   *Benefit:* Invariant to non-linear transformations (e.g., Log). Better for heavy-tailed risks.

### 1.3 Aggregation Levels

1.  **Level 1:** Within a Line of Business (e.g., Premium Risk + Reserve Risk).
2.  **Level 2:** Across Lines of Business (e.g., Motor + Property).
3.  **Level 3:** Across Risk Types (e.g., Underwriting + Market).
4.  **Level 4:** Across Legal Entities (Group Diversification).

---

## 2. Mathematical Framework

### 2.1 The Square Root Rule (Variance-Covariance)

$$ SCR_{total} = \sqrt{SCR^T \cdot R \cdot SCR} $$
*   $R$: Correlation Matrix.
*   *Assumption:* The tail risks are Elliptical.
*   *Limitation:* If risks are "Tail Dependent" (e.g., Pandemic + Financial Crash), this formula underestimates the capital.

### 2.2 Diversification Factor

$$ DF = \frac{SCR_{total}}{\sum SCR_i} $$
*   **Typical Values:**
    *   P&C Insurer: 60-70% (High diversification).
    *   Monoline Insurer: 90-95% (Low diversification).
    *   Composite (Life + Non-Life): 50-60% (Massive diversification).

---

## 3. Theoretical Properties

### 3.1 Granularity Effect

*   **Coarse:** Aggregating "All Non-Life" vs. "All Market".
    *   *Result:* Low diversification (because we implicitly assume 100% correlation inside the buckets).
*   **Granular:** Aggregating "Motor", "Fire", "Liability", "Equity", "Bond".
    *   *Result:* Higher diversification benefit.

### 3.2 The "Correlation Trap"

*   In normal times, Equity and Bonds might be negatively correlated (Flight to Quality).
*   In a liquidity crisis, they become positively correlated (Cash is King).
*   **Stress Testing:** Always test with "Stressed Correlations" (e.g., increase all correlations by 0.2).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Calculating Diversification (Python)

```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Inputs: Standalone Capital
risks = {
    'Motor': 100,
    'Property': 80,
    'Liability': 120,
    'Catastrophe': 150
}
scr_vec = np.array(list(risks.values()))

# Correlation Matrix
# Motor, Prop, Liab, Cat
corr = np.array([
    [1.0, 0.5, 0.2, 0.1], # Motor
    [0.5, 1.0, 0.1, 0.6], # Property (Corr with Cat)
    [0.2, 0.1, 1.0, 0.0], # Liability (Independent)
    [0.1, 0.6, 0.0, 1.0]  # Cat
])

# Aggregation
scr_total = np.sqrt(scr_vec.T @ corr @ scr_vec)
sum_scr = np.sum(scr_vec)
div_benefit = sum_scr - scr_total
div_factor = scr_total / sum_scr

print(f"Sum of SCRs: {sum_scr:.1f}")
print(f"Diversified SCR: {scr_total:.1f}")
print(f"Benefit: {div_benefit:.1f} ({div_benefit/sum_scr:.1%})")

# Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(corr, annot=True, xticklabels=risks.keys(), yticklabels=risks.keys(), cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()
```

### 4.2 Rank vs. Pearson

```python
from scipy.stats import spearmanr, pearsonr

# Generate data with non-linear dependence
x = np.random.normal(0, 1, 1000)
y = x**3 + np.random.normal(0, 1, 1000) # Cubic relationship

p_corr, _ = pearsonr(x, y)
s_corr, _ = spearmanr(x, y)

print(f"Pearson (Linear): {p_corr:.3f}")
print(f"Spearman (Rank): {s_corr:.3f}")
# Spearman should be higher because it captures the monotonic trend better.
```

---

## 5. Evaluation & Validation

### 5.1 Matrix Properties

*   **Positive Semi-Definite (PSD):** A valid correlation matrix must be PSD.
    *   *Check:* All eigenvalues must be $\ge 0$.
    *   *Issue:* If you "tweak" a correlation manually (e.g., change 0.2 to 0.8), you might break the PSD property.
    *   *Fix:* Use "Higham's Algorithm" to find the nearest PSD matrix.

### 5.2 Expert Judgement

*   "Why is the correlation between Cyber and D&O 0.5?"
*   **Validation:** Look for "Silent Cyber" scenarios where a hack leads to a lawsuit against Directors.
*   **Documentation:** Every correlation coefficient needs a justification.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Group Diversification**
    *   **Issue:** Holding capital at the Group level (Bermuda) for a risk in Germany.
    *   **Reality:** Capital is not "Fungible". Regulators might block the transfer of cash.
    *   **Solvency II:** "Group SCR" allows diversification, but "Solo SCR" does not.

2.  **Trap: Tail Dependence**
    *   **Issue:** Using Gaussian Copula for everything.
    *   **Reality:** Underestimates extreme joint events.

### 6.2 Implementation Challenges

1.  **Data Scarcity:**
    *   We don't have enough data to estimate the correlation between "Pandemic" and "Credit Default".
    *   **Solution:** Use Industry Benchmarks or Stress Scenarios.

---

## 7. Advanced Topics & Extensions

### 7.1 Hierarchical Aggregation

*   Instead of one giant matrix, we use a "Tree" structure.
    *   Step 1: Agg Market Risks.
    *   Step 2: Agg Insurance Risks.
    *   Step 3: Agg Market + Insurance.
*   *Benefit:* Easier to explain and manage.

### 7.2 Copula Aggregation (Internal Model)

*   Instead of the formula $\sqrt{xRx}$, we simulate.
*   Allows for non-linear dependencies.
*   **Result:** Usually gives a *lower* diversification benefit in the tail (because tails are thicker).

---

## 8. Regulatory & Governance Considerations

### 8.1 Standard Formula Correlations

*   Prescribed by EIOPA.
*   Example: Market vs. Non-Life = 0.25.
*   *Critique:* Too high? Too low? It's a compromise.

---

## 9. Practical Example

### 9.1 Worked Example: The "Merger"

**Scenario:**
*   Company A (Auto Insurer) merges with Company B (Life Insurer).
*   **Pre-Merger Capital:** \$100M + \$100M = \$200M.
*   **Post-Merger Capital:** \$150M.
*   **Synergy:** \$50M of capital released due to diversification.
*   **Value:** The merger creates value simply by optimizing the balance sheet.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Diversification** reduces capital.
2.  **Correlation** drives the benefit.
3.  **Granularity** matters.

### 10.2 When to Use This Knowledge
*   **M&A:** Valuing the capital synergies.
*   **Capital Allocation:** Deciding which line of business is "capital heavy".

### 10.3 Critical Success Factors
1.  **PSD Matrix:** Ensure the math works.
2.  **Stress Testing:** Don't believe the correlation is constant.
3.  **Fungibility:** Remember that cash is trapped in legal entities.

### 10.4 Further Reading
*   **Shaw:** "Multivariate Copulas in Actuarial Science".
*   **EIOPA:** "Correlations in the Standard Formula".

---

## Appendix

### A. Glossary
*   **PSD:** Positive Semi-Definite.
*   **Fungibility:** Ability to move capital freely.
*   **Granularity:** Level of detail.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Aggregated SCR** | $\sqrt{\sum \rho_{ij} SCR_i SCR_j}$ | Standard Formula |
| **Div Benefit** | $\sum SCR_i - SCR_{total}$ | KPI |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

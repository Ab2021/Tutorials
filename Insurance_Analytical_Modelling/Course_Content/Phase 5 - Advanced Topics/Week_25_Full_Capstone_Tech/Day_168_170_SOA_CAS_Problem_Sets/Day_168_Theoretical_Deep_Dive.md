# SOA/CAS-aligned Problem Sets (Part 1) - P&C and Health - Theoretical Deep Dive

## Overview
"The exams teach you the theory. Python teaches you the practice."
Actuarial exams (CAS Exam 5, SOA Exam FAM) are notorious for manual calculations.
In the real world, we don't use calculators; we use code.
This day bridges the gap. We take classic **Exam Problems** and solve them using **Python**, demonstrating how to scale the logic from 1 row to 1 million rows.

---

## 1. Conceptual Foundation

### 1.1 The "Parallelogram Method" (Ratemaking)

*   **Exam Logic:** Draw a geometric shape to calculate the "Earned Portion" of a rate change.
*   **Python Logic:** Vectorized date operations.
    *   `earned_premium = exposure * rate * (days_active / 365)`
*   **Why it matters:** The geometric method breaks down when you have daily policy writing. Python handles it natively.

### 1.2 The Chain Ladder Method (Reserving)

*   **Exam Logic:** Calculate Link Ratios ($f_{12}, f_{23}$) by hand. Select the average.
*   **Python Logic:** `df.groupby('dev_period').sum()` to get the triangle. Then `f_12 = sum(c_12) / sum(c_11)`.
*   **Scale:** Exams give you 5 accident years. Python handles 50 years x 50 states.

---

## 2. Mathematical Framework

### 2.1 On-Leveling Premiums

To price future policies, we must adjust past premiums to current rate levels.
$$ P_{on-level} = P_{historical} \times \prod (1 + \Delta Rate) $$
*   **Challenge:** Policies are written throughout the year. A rate change on July 1st affects only 50% of the exposure for a Jan 1st policy.

### 2.2 Health PMPM (Per Member Per Month)

$$ PMPM = \frac{\text{Total Claims Cost}}{\text{Member Months}} $$
*   **Utilization:** Visits / 1000 Members.
*   **Unit Cost:** Cost / Visit.
*   **Equation:** $PMPM = \text{Utilization} \times \text{Unit Cost} / 12000$.

---

## 3. Theoretical Properties

### 3.1 Credibility-Weighted LDFs

*   **Exam 5:** You might be asked to weight the "Volume-Weighted Average" vs. the "Simple Average".
*   **Python:** We can simulate 10,000 triangles to see which average performs better (Bootstrapping).

### 3.2 Bornhuetter-Ferguson (BF)

*   **Formula:** $R = \text{Paid} + (1 - \frac{1}{LDF}) \times \text{Expected Loss}$.
*   **Python Implementation:**
    *   Inputs: `paid_vector`, `ldf_vector`, `apriori_lr`, `earned_premium`.
    *   Output: `reserve_vector`.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Problem 1: The Parallelogram Method (Pythonized)

**Problem:** Rates increased 5% on July 1, 2022. Calculate the On-Level Factor for CY 2022 Earned Premium.

```python
import pandas as pd
import numpy as np

# Simulation: 10,000 Policies written uniformly in 2022
n_policies = 10000
dates = pd.date_range(start='2022-01-01', end='2022-12-31', periods=n_policies)
policies = pd.DataFrame({'policy_start': dates})

# Rate Change Date
rate_change_date = pd.Timestamp('2022-07-01')

# Function to calculate effective rate for a policy
def get_rate_index(start_date):
    # Policy lasts 1 year.
    # Portion before July 1: Rate = 1.00
    # Portion after July 1: Rate = 1.05
    end_date = start_date + pd.Timedelta(days=365)
    
    if start_date >= rate_change_date:
        return 1.05
    elif end_date <= rate_change_date:
        return 1.00
    else:
        # Split policy
        days_pre = (rate_change_date - start_date).days
        days_post = 365 - days_pre
        weighted_rate = (days_pre * 1.00 + days_post * 1.05) / 365
        return weighted_rate

policies['rate_index'] = policies['policy_start'].apply(get_rate_index)

# On-Level Factor = Current Rate (1.05) / Average Earned Rate
olf = 1.05 / policies['rate_index'].mean()
print(f"Calculated OLF: {olf:.4f}")
# Theoretical Geometric Answer: 1.05 / (1.00 * 0.125 + 1.025 * 0.75 + 1.05 * 0.125) approx.
```

### 4.2 Problem 2: Reserving with Bootstrapping

**Problem:** Given a triangle, calculate the range of reasonable reserves.

```python
# Triangle Data (Cumulative Paid)
triangle = np.array([
    [100, 150, 175, 180],
    [110, 160, 190, 0],
    [120, 170, 0, 0],
    [130, 0, 0, 0]
])

def chain_ladder(tri):
    # Calculate LDFs
    ldfs = []
    for col in range(tri.shape[1] - 1):
        # Only use rows where next column is non-zero
        mask = tri[:, col+1] > 0
        if sum(mask) == 0: break
        ldf = tri[mask, col+1].sum() / tri[mask, col].sum()
        ldfs.append(ldf)
    return ldfs

# Bootstrapping (Simplified ODP)
# In practice, we would calculate residuals and resample them.
# Here, we just perturb the LDFs slightly for demonstration.
simulated_reserves = []
base_ldfs = chain_ladder(triangle)

for _ in range(1000):
    # Randomize LDFs
    sim_ldfs = [l * np.random.normal(1, 0.05) for l in base_ldfs]
    # Project Ultimate
    # (Implementation of projection logic omitted for brevity)
    pass 
```

---

## 5. Evaluation & Validation

### 5.1 Checking against "The Book"

*   **Validation:** Does your Python code match the example in the Werner & Modlin textbook?
*   **Discrepancy:** Often, Python is *more* accurate because it doesn't round intermediate steps.
*   **Action:** Replicate the rounding behavior (`np.round(x, 3)`) to match the exam exactly, then remove it for production.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Calendar-Year" Trap

*   **Issue:** In Ratemaking, we use "Policy Year" or "Calendar-Accident Year".
*   **Exam Trick:** Giving you Calendar Year data and asking for Policy Year rate need.
*   **Python:** `df.groupby(df['policy_start'].dt.year)` vs `df.groupby(df['loss_date'].dt.year)`. Be explicit.

### 6.2 Tail Factors

*   **Exam:** "Select a tail factor of 1.02."
*   **Reality:** Fitting a curve (Inverse Power, Exponential) to the LDFs to extrapolate the tail.
    *   `scipy.optimize.curve_fit`.

---

## 7. Advanced Topics & Extensions

### 7.1 Reinsurance Pricing (Exposure Rating)

*   **Problem:** Price a \$500k xs \$500k layer using a Severity Curve.
*   **Python:**
    *   `def limited_expected_value(limit): return integrate(x * pdf(x), 0, limit) + limit * (1 - cdf(limit))`
    *   `layer_cost = limited_expected_value(1M) - limited_expected_value(500k)`

### 7.2 Health Risk Adjustment

*   **Problem:** Calculate the relative risk score of a population.
*   **Python:** Merge membership data with HCC (Hierarchical Condition Category) weights.
    *   `df.merge(hcc_table, on='diagnosis_code')`.

---

## 8. Regulatory & Governance Considerations

### 8.1 ASOP (Actuarial Standards of Practice)

*   **ASOP 23 (Data Quality):** You must review the data for reasonableness.
*   **Python Check:** `assert df['paid_loss'].min() >= 0` (unless recoveries exceed payments).

---

## 9. Practical Example

### 9.1 The "Exam 5" Project

**Scenario:** You are studying for Exam 5.
**Task:** Instead of doing the practice problems on paper, write a Python script for each one.
**Benefit:**
1.  You learn the material deeper.
2.  You build a library of code snippets you can use at work.
3.  You pass the exam.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Parallelogram Method** = Date Math.
2.  **Chain Ladder** = Groupby + Sum.
3.  **PMPM** = Weighted Averages.

### 10.2 When to Use This Knowledge
*   **Exams:** To visualize the problem.
*   **Work:** To implement the theory efficiently.

### 10.3 Critical Success Factors
1.  **Vectorization:** Don't use `for` loops over policies. Use `pandas`.
2.  **Visualization:** Plot the triangles. It helps intuition.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking".
*   **Friedland:** "Estimating Unpaid Claims Using Basic Techniques".

---

## Appendix

### A. Glossary
*   **OLF:** On-Level Factor.
*   **LDF:** Loss Development Factor.
*   **CY:** Calendar Year.
*   **PY:** Policy Year.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Chain Ladder** | $U = C \times \prod LDF$ | Reserving |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

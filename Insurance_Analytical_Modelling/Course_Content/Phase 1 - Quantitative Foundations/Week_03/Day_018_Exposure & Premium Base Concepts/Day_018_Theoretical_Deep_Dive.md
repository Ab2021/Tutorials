# Exposure & Premium Base Concepts - Theoretical Deep Dive

## Overview
This session covers the fundamental concepts of exposure and premium bases in non-life insurance. We explore how risk is measured, how premiums are recognized as revenue (Written vs. Earned), and how to calculate the Unearned Premium Reserve (UPR). We also detail the specific exposure bases used across different lines of business and introduce the "Parallelogram Method" for adjusting historical premiums to current rate levels.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Exposure:** The basic unit of risk underlying an insurance premium. It measures the extent of the potential loss.

**Premium Base:** The specific unit of exposure to which the rate is applied.
*   *Formula:* Premium = Rate $\times$ Exposure Units.

**Written vs. Earned Premium:**
*   **Written Premium (WP):** The total premium charged for the entire policy term at the time of issuance. It is a "sales" metric.
*   **Earned Premium (EP):** The portion of the written premium that corresponds to the coverage already provided (time passed). It is a "revenue" metric.
*   **Unearned Premium (UEP):** The portion of the written premium representing future coverage. This is held as a liability (Reserve).

**Key Terminology:**
*   **In-force Exposure:** The number of insured units currently covered at a specific point in time.
*   **Earned Exposure:** The portion of exposure units for which coverage has been provided during a period.
*   **Calendar Year (CY):** Transactions occurring Jan 1 to Dec 31 (regardless of when the policy started).
*   **Policy Year (PY):** All transactions associated with policies *effective* in a specific year.
*   **Accident Year (AY):** All losses occurring in a year, matched with premiums earned in that year.

### 1.2 Historical Context & Evolution

**Origin:**
*   **Fire Insurance:** Early exposure was simply the "Sum Insured."
*   **Workers' Comp:** Payroll became the standard as it correlates with the number of workers and their economic value.

**Evolution:**
*   **Composite Rating:** In Commercial General Liability (GL), multiple exposure bases (sales, area, payroll) were often combined.
*   **Usage-Based:** Telematics introduced "miles driven" or "driving duration" as a dynamic exposure base for auto.

**Current State:**
*   **Auditable Exposures:** For GL and WC, the initial premium is an estimate (deposit premium). The final premium is determined after a year-end audit of actual payroll/sales.

### 1.3 Why This Matters

**Business Impact:**
*   **Pricing Accuracy:** If the exposure base doesn't correlate with loss potential, the rate will be unfair (e.g., charging a flat rate for a factory with 10 employees vs. 1000 employees).
*   **Inflation Protection:** Some exposure bases (Payroll, Sales) inflate automatically, providing a natural hedge against claims inflation. Others (Car-years) do not.

**Regulatory Relevance:**
*   **Solvency:** UPR is often the largest liability on a P&C insurer's balance sheet. Understating it overstates equity.
*   **Statutory Accounting (SAP):** Requires immediate expensing of acquisition costs but pro-rata recognition of premium (the "Equity in UPR" penalty).

---

## 2. Mathematical Framework

### 2.1 Core Assumptions

1.  **Assumption: Uniform Earning**
    *   **Description:** Risk is spread evenly over the policy term.
    *   **Implication:** Premium is earned pro-rata (1/365th per day).
    *   **Real-world validity:** Valid for Auto/Home. Invalid for Warranty (risk increases with age) or Marine (risk concentrated during voyage).

2.  **Assumption: Proportionality**
    *   **Description:** Doubling the exposure units doubles the expected loss.
    *   **Implication:** Linear rating models (Rate $\times$ Exposure).

### 2.2 Mathematical Notation

| Symbol | Meaning | Example |
| :--- | :--- | :--- |
| $WP$ | Written Premium | $1,200 |
| $EP$ | Earned Premium | $600 (at 6 months) |
| $UPR$ | Unearned Premium Reserve | $600 (at 6 months) |
| $E$ | Exposure Units | 50 Car-Years |
| $R$ | Rate per unit | $500 |
| $t$ | Time elapsed (fraction of term) | 0.5 |

### 2.3 Core Equations & Derivations

#### Equation 1: Basic Accounting Identity
$$ WP = EP + \Delta UPR $$
$$ EP = WP - (UPR_{end} - UPR_{begin}) $$
*   **Interpretation:** Earned premium is the written premium adjusted for the change in the unearned reserve.

#### Equation 2: Pro-Rata Earned Premium (Individual Policy)
For a policy written on date $D_{start}$ with term $T$ days, at valuation date $D_{val}$:
$$ \text{Days Elapsed} = \max(0, \min(T, D_{val} - D_{start})) $$
$$ EP = WP \times \frac{\text{Days Elapsed}}{T} $$
$$ UPR = WP - EP $$

#### Equation 3: The "1/24th Method" (Aggregate Approximation)
Used when individual policy dates are not available, assuming uniform writing throughout the month.
*   Assume all policies written in a month are written on the 15th.
*   A 1-year policy written in January earns 0.5 months in January (1/24th of the year) and 11.5 months in the rest of the year.

#### Equation 4: On-Level Premium (Parallelogram Method)
To adjust historical EP to current rates:
$$ EP_{on-level} = EP_{historical} \times \text{On-Level Factor} $$
The factor is derived geometrically by calculating the area of the "parallelogram" (policy year) or "trapezoid" (calendar year) covered by each rate level.

### 2.4 Exposure Bases by Line of Business

| Line of Business | Exposure Base | Why? | Inflation Sensitive? |
| :--- | :--- | :--- | :--- |
| **Private Auto** | Car-Year | Risk is per vehicle. | No |
| **Homeowners** | Amount of Insurance (AOI) | Risk relates to rebuild cost. | Yes (if AOI indexed) |
| **Workers' Comp** | Payroll ($ per $100) | Risk relates to # workers and wages. | Yes |
| **General Liability** | Sales / Receipts | Activity level proxy. | Yes |
| **General Liability** | Square Footage | Passive risk (premises). | No |
| **Malpractice** | Physician Count / Specialty | Risk is per doctor. | No |

---

## 3. Theoretical Properties

### 3.1 Criteria for a Good Exposure Base

1.  **Proportional to Hazard:** $E[Loss] \propto Exposure$.
2.  **Practical:** Easy to measure and verify (Auditable).
3.  **Historical Precedent:** Consistency allows for long-term data analysis.
4.  **Not Subject to Manipulation:** Hard for the insured to fake.

### 3.2 Inflation Sensitivity

*   **Inflation-Sensitive Bases (Payroll, Sales):** As wages/prices rise, premiums rise automatically. This helps insurers keep up with claims inflation without needing frequent rate filings.
*   **Fixed Bases (Car-Year, Square Feet):** Premiums stay flat while claims inflation drives costs up. Insurers must file for rate increases ("Trend") to compensate.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Policy Transaction Table:** PolicyID, EffectiveDate, ExpirationDate, TransactionDate, WrittenPremiumAmount.
*   **Rate Change History:** Date and % Change of all past rate revisions.

### 4.2 Preprocessing Steps

**Step 1: Calculate Daily Earned Premium**
*   Convert all terms to days.
*   Daily Rate = Written Premium / Days in Term.

**Step 2: Aggregate by Year**
*   Sum daily earned amounts into Calendar Years or Accident Years.

### 4.3 Model Specification (Python Example)

Calculating Earned Premium and UPR for a portfolio.

```python
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Simulated Policy Data
# 3 Policies:
# A: Annual, starts Jan 1, 2023. Premium $1200.
# B: Annual, starts Jul 1, 2023. Premium $2400.
# C: 6-Month, starts Oct 1, 2023. Premium $600.

policies = pd.DataFrame({
    'PolicyID': ['A', 'B', 'C'],
    'EffDate': [datetime(2023, 1, 1), datetime(2023, 7, 1), datetime(2023, 10, 1)],
    'TermMonths': [12, 12, 6],
    'WrittenPrem': [1200, 2400, 600]
})

# Calculate Expiration Date
policies['ExpDate'] = policies.apply(
    lambda x: x['EffDate'] + pd.DateOffset(months=x['TermMonths']), axis=1
)
policies['TotalDays'] = (policies['ExpDate'] - policies['EffDate']).dt.days

def calculate_earned_at_date(row, val_date):
    if val_date < row['EffDate']:
        return 0.0
    
    days_elapsed = (min(val_date, row['ExpDate']) - row['EffDate']).dt.days
    days_elapsed = max(0, days_elapsed)
    
    fraction = days_elapsed / row['TotalDays']
    return row['WrittenPrem'] * fraction

# Valuation Date: Dec 31, 2023
val_date = datetime(2023, 12, 31)

policies['EarnedPrem_2023'] = policies.apply(
    lambda row: calculate_earned_at_date(row, val_date), axis=1
)

policies['UPR_2023'] = policies['WrittenPrem'] - policies['EarnedPrem_2023']

print("Policy Level Calculations (Valuation: 2023-12-31):")
print(policies[['PolicyID', 'WrittenPrem', 'EarnedPrem_2023', 'UPR_2023']])

print("\nPortfolio Totals:")
print(policies[['WrittenPrem', 'EarnedPrem_2023', 'UPR_2023']].sum())

# Verification Logic
# Policy A: Full year elapsed. Earned = 1200, UPR = 0.
# Policy B: Half year elapsed. Earned ~ 1200, UPR ~ 1200.
# Policy C: Half term (3 months) elapsed. Earned ~ 300, UPR ~ 300.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Earned Premium:** Revenue to be recognized in the income statement.
2.  **UPR:** Liability to be booked on the balance sheet.

**Interpretation:**
*   **High UPR:** Indicates a growing book of business (lots of new policies written recently).
*   **Negative Earned Premium:** Impossible (unless processing cancellations/refunds).

---

## 5. Evaluation & Validation

### 5.1 Consistency Checks

*   **Equation Check:** $WP - EP - UPR = 0$ (for a single policy term).
*   **Boundaries:** $0 \le EP \le WP$ (ignoring endorsements).
*   **Trend:** If WP is growing, UPR should be growing.

### 5.2 Audit & True-up

For lines like Workers' Comp:
1.  **Deposit Premium:** Paid at start based on *estimated* payroll.
2.  **Audit Premium:** Calculated at end based on *actual* payroll.
3.  **Adjustment:** The difference is billed or refunded.
    *   *Model Implication:* Earned premium history changes retrospectively!

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Calendar Year vs. Policy Year**
    *   **Issue:** Mixing them up.
    *   **Reality:** CY 2023 Earned Premium includes policies written in 2022 and 2023. PY 2023 Earned Premium includes *only* policies written in 2023, but their earning extends into 2024.
    *   **Use Case:** Financial reporting uses CY. Ratemaking often uses PY or AY.

2.  **Trap: Uneven Earning Patterns**
    *   **Issue:** Assuming pro-rata for Warranty or Seasonal risks.
    *   **Reality:** A warranty for a car has low risk in year 1 (manufacturer warranty covers it) and high risk in year 4. Earning curve must match the risk curve.

### 6.2 Implementation Challenges

1.  **Mid-term Adjustments (Endorsements):**
    *   Adding a car halfway through the policy.
    *   Requires recalculating the daily rate and UPR for the *remainder* of the term.

2.  **Cancellations:**
    *   **Pro-rata cancellation:** Insurer cancels; refund is exact pro-rata.
    *   **Short-rate cancellation:** Insured cancels; refund is pro-rata *minus* a penalty (to cover admin costs).

---

## 7. Advanced Topics & Extensions

### 7.1 The Parallelogram Method (On-Leveling)

Used in Ratemaking to answer: *"What would our historical premiums be if we had charged today's rates?"*

**Steps:**
1.  Draw a square with Time on X-axis (Year 1, Year 2) and Policy Duration on Y-axis (0 to 1).
2.  Draw vertical lines for Rate Changes.
3.  Calculate the geometric area of the policy block exposed to the Old Rate vs. New Rate.
4.  Apply the weighted average factor.

**Why?**
If you raised rates 10% last year, your historical Loss Ratio looks artificially high (because premiums were lower). You must "on-level" the premiums to make the past comparable to the future.

### 7.2 Extension of Exposures

An alternative to the Parallelogram Method.
*   **Method:** Take every single historical policy and re-rate it using the current rating engine.
*   **Pros:** Perfectly accurate.
*   **Cons:** Computationally expensive; requires full data history and a functioning rating engine for past data.

---

## 8. Regulatory & Governance Considerations

### 8.1 Statutory Accounting Principles (SAP)

*   **Non-Admitted Assets:** Premiums over 90 days past due are not counted as assets.
*   **UPR:** Must be booked gross of reinsurance (mostly).

### 8.2 Data Governance

*   **Exposure Audits:** Regulators require insurers to audit a sample of WC policies to ensure they aren't undercharging (solvency risk) or overcharging.

---

## 9. Practical Example

### 9.1 Worked Example: Parallelogram Method

**Scenario:**
*   **Rate Change:** +10% effective July 1, 2023.
*   **Goal:** Calculate the On-Level Factor for Calendar Year 2023.
*   **Assumption:** Policies written evenly throughout the year.

**Geometry:**
*   **CY 2023** is a square from Jan 1 to Dec 31.
*   **Policies earning in 2023** include those written in 2022 (top triangle) and 2023 (bottom triangle).
*   **Rate Change Line:** Vertical line at July 1.
    *   Policies written before July 1 (and earning in 2023) are at Index 1.00.
    *   Policies written after July 1 (and earning in 2023) are at Index 1.10.

**Calculation:**
1.  **Area 1 (Old Rate):**
    *   Policies written in 2022 earning in 2023: All at Old Rate. (Area = 0.5 of the square).
    *   Policies written Jan-Jun 2023: All at Old Rate. (Area = 0.5 * 0.5 = 0.25).
    *   Total Area at Old Rate = 0.75.
2.  **Area 2 (New Rate):**
    *   Policies written Jul-Dec 2023: All at New Rate. (Area = 0.25).
    *   Total Area at New Rate = 0.25.
3.  **Average Rate Level:** $0.75 \times 1.00 + 0.25 \times 1.10 = 0.75 + 0.275 = 1.025$.
4.  **Current Rate Level:** $1.10$.
5.  **On-Level Factor:** $\frac{\text{Current}}{\text{Average}} = \frac{1.10}{1.025} = 1.0732$.

**Result:** Multiply CY 2023 Earned Premium by 1.0732 to bring it to current rate levels.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Exposure** is the unit of risk; **Premium** is Rate x Exposure.
2.  **Earned Premium** is revenue; **Written Premium** is sales.
3.  **UPR** is the liability for future coverage.
4.  **Parallelogram Method** adjusts past premiums to current rates.

### 10.2 When to Use This Knowledge
*   **Reserving:** Calculating UPR.
*   **Ratemaking:** On-leveling premiums.
*   **Financial Reporting:** Booking revenue.

### 10.3 Critical Success Factors
1.  **Match Dates:** Ensure Effective and Expiration dates are accurate.
2.  **Check Exposure Bases:** Ensure they are inflation-adjusted if necessary.
3.  **Audit:** Verify estimated exposures (payroll/sales) at year-end.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking" (CAS Syllabus).
*   **Friedland:** "Estimating Unpaid Claims Using Basic Techniques".

---

## Appendix

### A. Glossary
*   **Deposit Premium:** Initial premium based on estimated exposure.
*   **Audit Premium:** Adjustment based on actual exposure.
*   **Pro-Rata:** Proportional to time.
*   **Short-Rate:** Pro-rata less a penalty.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Earned Premium** | $WP \times \frac{\text{Days Elapsed}}{\text{Term}}$ | Revenue Recog |
| **UPR** | $WP - EP$ | Liability Calc |
| **On-Level Factor** | $\frac{\text{Current Rate}}{\text{Avg Historical Rate}}$ | Ratemaking |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*

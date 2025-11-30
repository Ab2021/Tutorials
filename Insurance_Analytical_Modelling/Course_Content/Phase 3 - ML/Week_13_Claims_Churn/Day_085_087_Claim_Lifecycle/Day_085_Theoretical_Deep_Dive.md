# Claim Lifecycle & Severity Development (Part 1) - The Lifecycle Framework - Theoretical Deep Dive

## Overview
"A claim is a living organism."
It is born (Occurrence), reported (FNOL), grows (Reserving), eats cash (Payments), and eventually dies (Settlement).
Understanding this lifecycle is crucial for **Individual Claim Reserving** and **Operational Efficiency**.
This day focuses on modeling the *states* and *transitions* of a claim's life.

---

## 1. Conceptual Foundation

### 1.1 The Timeline of a Claim

1.  **Occurrence Date ($t_0$):** The accident happens.
2.  **Report Date ($t_1$):** The insurer finds out (FNOL).
    *   $t_1 - t_0$ = **Reporting Delay**.
3.  **Payment Dates ($t_2, t_3...$):** Partial payments (Indemnity, Medical, Expense).
4.  **Closure Date ($t_n$):** The claim is settled.
5.  **Reopening Date:** Sometimes, it comes back to life (Relapse).

### 1.2 Severity Development

*   **Initial Reserve:** Set by adjuster at FNOL (often a "Case Estimate").
*   **Development:** As facts emerge (Medical report, Lawyer letter), the reserve changes.
*   **Stair-stepping:** The tendency for reserves to jump up in discrete steps rather than smooth increases.

---

## 2. Mathematical Framework

### 2.1 Multi-State Models

*   **States:**
    *   0: Not Reported (IBNR).
    *   1: Reported / Open.
    *   2: Closed with Payment.
    *   3: Closed without Payment (CWP).
*   **Transitions:**
    *   $0 \to 1$ (Reporting).
    *   $1 \to 2$ (Settlement).
    *   $2 \to 1$ (Reopening).
*   **Math:** Continuous Time Markov Chains (CTMC) or Semi-Markov Models (if duration matters).

### 2.2 Reporting Delay Distribution

*   **Model:** Weibull or Lognormal.
*   **Formula:** $P(T \le t) = 1 - e^{-(t/\lambda)^k}$.
*   **Insight:** Long tails in reporting delay (e.g., Asbestos) drive IBNR.

---

## 3. Theoretical Properties

### 3.1 The "RBNS" vs "IBNR" Split

*   **RBNS (Reported But Not Settled):** We know the claim exists, but not the final cost.
    *   Driven by *Severity Development*.
*   **IBNR (Incurred But Not Reported):** We don't know the claim exists.
    *   Driven by *Reporting Delay*.
*   **Lifecycle Model:** Explicitly separates these two processes, unlike Chain Ladder which lumps them.

### 3.2 Operational Bottlenecks

*   **Queue Theory:** Claims are customers in a queue waiting for adjusters.
*   **Metric:** "Days to Initial Contact".
*   **Impact:** Longer queue time $\to$ Higher Severity (Litigation risk increases).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Survival Analysis for Settlement (Python lifelines)

```python
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

# Data: Duration (Days open), Event (1=Closed, 0=Open)
T = df['duration']
E = df['is_closed']

kmf = KaplanMeierFitter()
kmf.fit(T, event_observed=E)

# Plot Survival Curve (Prob of remaining open)
kmf.plot_survival_function()
plt.title("Claim Lifespan Analysis")
plt.xlabel("Days since Report")
plt.ylabel("Probability Open")
```

### 4.2 Transition Matrix Estimation

```python
# Count transitions between states per month
# Matrix P
#       Open   Closed   Reopen
# Open  0.90   0.09     0.00
# Closed 0.01   0.99     0.00

# Prediction
# State_Vector_t+1 = State_Vector_t * P
```

---

## 5. Evaluation & Validation

### 5.1 Duration Accuracy

*   **Metric:** Mean Absolute Error (MAE) of predicted vs. actual days to close.
*   **Use:** Workload planning. "We expect 500 claims to close next month, freeing up 10 adjusters."

### 5.2 Reserve Adequacy

*   **Check:** Compare "Case Reserve" vs. "Predicted Ultimate" from the Lifecycle Model.
*   **Gap:** If Model > Case, the adjusters are under-reserving.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Censoring

*   **Issue:** Most recent claims are still open (Right Censored).
*   **Mistake:** Dropping open claims and calculating "Average Cost of Closed Claims".
*   **Result:** Massive underestimation (because small claims close fast, big claims stay open).
*   **Fix:** Use Survival Analysis (Kaplan-Meier) which handles censoring correctly.

### 6.2 Reopenings

*   **Issue:** A "Closed" claim isn't always closed.
*   **Impact:** If you ignore reopenings, you underestimate the tail liability.
*   **Model:** Probability(Reopen | Time since Close).

---

## 7. Advanced Topics & Extensions

### 7.1 Granular Lifecycle (Granular Models)

*   **Idea:** Model specific events: "Surgery", "Legal Suit", "Investigation".
*   **Process Mining:** Use logs from the Claims Management System (Guidewire) to visualize the actual process flow.
    *   "Why do 30% of claims loop back from 'Payment' to 'Investigation'?"

### 7.2 Behavioral Modeling

*   **Agent:** Model the *Adjuster's* behavior.
*   **Bias:** "Adjuster Bob always closes claims on Fridays."
*   **Optimization:** Nudge adjusters to follow the optimal path.

---

## 8. Regulatory & Governance Considerations

### 8.1 Prompt Payment Laws

*   **Regulation:** Insurers must pay clean claims within X days (e.g., 30 days in Texas).
*   **Monitoring:** Use the Lifecycle Model to predict "Breach Probability" and alert management.

---

## 9. Practical Example

### 9.1 The "Fast Track" Algorithm

**Scenario:** High volume of windshield claims.
**Model:** Lifecycle prediction.
**Rule:**
*   If Predicted Severity < \$500 AND Predicted Complexity = Low:
    *   **Auto-Adjudicate:** Pay immediately without human review.
*   **Result:** 60% of claims handled by bot. Adjusters focus on the complex 40%.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **States & Transitions** define the lifecycle.
2.  **Censoring** must be handled statistically.
3.  **Reporting Delay** drives IBNR.

### 10.2 When to Use This Knowledge
*   **Operational Excellence:** Reducing cycle time.
*   **Reserving:** Micro-reserving (Individual Claim Reserving).

### 10.3 Critical Success Factors
1.  **Data Granularity:** You need daily/weekly snapshots of the claim status, not just quarterly triangles.
2.  **Process Knowledge:** Understand *why* a claim moves from A to B.

### 10.4 Further Reading
*   **Taylor, McGuire:** "GLM for Loss Reserving II: Structure and Lifecycle".
*   **Aaltonen:** "Process Mining in Insurance".

---

## Appendix

### A. Glossary
*   **CWP:** Closed Without Payment (Denial or Withdrawal).
*   **Salvage:** Recovery of value from damaged property (e.g., selling the wrecked car).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Survival Function** | $S(t) = P(T > t)$ | Duration Modeling |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

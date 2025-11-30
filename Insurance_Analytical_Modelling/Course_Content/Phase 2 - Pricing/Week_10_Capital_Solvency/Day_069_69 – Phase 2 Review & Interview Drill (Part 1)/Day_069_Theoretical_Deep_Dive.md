# Phase 2 Review & Interview Drill (Part 1) - Theoretical Deep Dive

## Overview
You have learned the math (Chain Ladder, GLM, BF). Now you must learn to explain it. This session focuses on **Pricing & Reserving Interview Questions**. We cover **Behavioral**, **Technical**, and **Case Study** questions.

---

## 1. Behavioral Questions (The "Fit" Check)

### 1.1 "Tell me about a time you explained a complex technical concept to a non-technical audience."

*   **Bad Answer:** "I explained the GLM formula to the Underwriter."
*   **Good Answer (STAR Method):**
    *   **Situation:** The Underwriter wanted to price a large fleet but didn't trust the model.
    *   **Task:** I needed to explain why the model suggested a 20% rate increase.
    *   **Action:** Instead of showing coefficients, I showed a "Base Rate + Add-ons" table. I explained that "Young Drivers" were the driver of the increase.
    *   **Result:** The Underwriter accepted the price and we won the account.

### 1.2 "Describe a time you made a mistake."

*   **Goal:** Show integrity and resilience.
*   **Key:** Admit the mistake quickly, fix it, and implement a control to prevent recurrence.

---

## 2. Technical Questions (The "Knowledge" Check)

### 2.1 Reserving

**Q: What is the difference between Chain Ladder and Bornhuetter-Ferguson?**
*   **Answer:**
    *   **Chain Ladder:** Relies entirely on the data in the triangle. Good for stable, mature lines. Volatile for green years.
    *   **BF:** Blends the data with an *a priori* expectation (Loss Ratio). Good for immature years where the triangle is sparse.

**Q: Why might the Paid LDF be higher than the Incurred LDF?**
*   **Answer:**
    *   **Case Reserve Strengthening:** If case reserves are increasing faster than payments, Incurred LDFs will be high.
    *   **Speed of Settlement:** If claims are settling faster (Paid accelerates), Paid LDFs drop. If they settle slower, Paid LDFs rise.
    *   *Trick:* Usually, Incurred LDFs < Paid LDFs because Incurred is closer to Ultimate.

**Q: How do you handle negative incremental losses in a triangle?**
*   **Answer:**
    *   **Cause:** Salvage/Subrogation recoveries.
    *   **Impact:** Chain Ladder fails (Log of negative number).
    *   **Fix:** Use the Mack method (additive) or set the LDF to 1.0 and handle manually.

### 2.2 Pricing

**Q: Explain a GLM to a 5-year-old.**
*   **Answer:** "Imagine a pizza. The base price is \$10. If you add pepperoni (Young Driver), add \$2. If you add mushrooms (Urban Area), add \$1. The final price is the sum of the toppings." (Additive model).
*   *Note:* Real GLMs are multiplicative ($10 \times 1.2 \times 1.1$), but the analogy holds.

**Q: What is the "One-Way Analysis" in GLM modeling?**
*   **Answer:** Plotting the Average Loss vs. a single variable (e.g., Age). It helps check if the relationship is linear, monotonic, or U-shaped.

---

## 3. Case Study Drills (The "Application" Check)

### 3.1 Case 1: The "Exploding" Loss Ratio

**Scenario:** You are the Reserving Actuary. The CFO runs into your office. "The Loss Ratio for the current accident year just jumped from 60% to 80% in one quarter! Why?"

**Investigation Steps:**
1.  **Data Error:** Did we double-count a file? (Check claim counts).
2.  **Large Loss:** Is it one massive claim (Shock) or many small ones (Trend)?
3.  **Speedup:** Did the Claims Department speed up payments? (Check Paid vs. Incurred).
4.  **Premium:** Did the denominator (Earned Premium) drop?
5.  **Mix:** Did we write a new, risky class of business?

### 3.2 Case 2: The "Undercutting" Competitor

**Scenario:** You are the Pricing Actuary. Sales says, "Competitor X is 20% cheaper than us for young drivers. We are losing all our volume."

**Analysis:**
1.  **Profitability:** Are we making money at our *current* price? If yes, maybe we can cut. If no, let them have the volume (Winner's Curse).
2.  **Segmentation:** Maybe Competitor X has a better model (Telematics?) and is picking off the *good* young drivers.
3.  **Strategy:** Do we want to be in this market?

---

## 4. Modeling Artifacts & Implementation

### 4.1 The "Mock Interview" Script

*   **Interviewer:** "How do you select the Tail Factor?"
*   **Candidate:** "I look at:
    1.  **Industry Benchmarks:** What does the 'Yellow Book' say?
    2.  **Curve Fitting:** Fitting an Inverse Power Curve to the LDFs.
    3.  **Judgment:** Is there latent liability (Asbestos)?"

### 4.2 Python Drill: Curve Fitting for Tail Factors

```python
import numpy as np
from scipy.optimize import curve_fit

# Data: Development Years (x) and LDFs (y)
x_data = np.array([1, 2, 3, 4, 5])
y_data = np.array([1.5, 1.2, 1.1, 1.05, 1.02])

# Model: Inverse Power Curve: y = 1 + a * x^(-b)
def inverse_power(x, a, b):
    return 1 + a * x**(-b)

# Fit
popt, pcov = curve_fit(inverse_power, x_data, y_data)
a_opt, b_opt = popt

# Extrapolate to Year 10
tail_ldf = inverse_power(10, a_opt, b_opt)
print(f"Fitted Parameters: a={a_opt:.2f}, b={b_opt:.2f}")
print(f"Projected LDF at Year 10: {tail_ldf:.3f}")
```

---

## 5. Evaluation & Validation

### 5.1 The "So What?" Test

*   Every answer must have a "So What?"
*   *Example:* "I calculated the LDFs." (So what?) -> "This allowed me to reduce the IBNR by \$5M, which increased Net Income."

### 5.2 Communication Clarity

*   Avoid jargon when talking to non-actuaries.
*   Use "Reserves" instead of "IBNR".
*   Use "Price" instead of "Rate per Exposure".

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Over-Confidence**
    *   **Q:** "Is your reserve estimate correct?"
    *   **A:** "No. It is an estimate. The actual result will differ. Here is the range."

2.  **Trap: Ignoring the Business**
    *   **Q:** "Why did you pick the 5-year average?"
    *   **A:** "Because the math said so." (Bad).
    *   **A:** "Because the claims handling changed 5 years ago, so older data is irrelevant." (Good).

---

## 7. Advanced Topics & Extensions

### 7.1 Machine Learning in Interviews

*   **Q:** "How would you use GBM (Gradient Boosting) in pricing?"
*   **A:** "I would use it to find non-linear interactions (e.g., Age * Vehicle Power), then feed those interactions back into a GLM for regulatory approval."

---

## 8. Regulatory & Governance Considerations

### 8.1 Professionalism

*   **ASOPs (Actuarial Standards of Practice):** Mentioning ASOP 43 (Reserving) or ASOP 12 (Risk Classification) shows you are a professional.

---

## 9. Practical Example

### 9.1 Worked Example: The "Elevator Pitch"

**Scenario:** You are in an elevator with the CEO.
**CEO:** "How are the reserves looking?"
**You:** "Stable. We saw some pressure in Auto Liability due to inflation, but we released some redundancy from the 2020 year to offset it. Overall, we are within the 5% margin of safety."
**Result:** Concise, high-level, reassuring.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **STAR Method** for behavioral.
2.  **First Principles** for technical.
3.  **Business Context** for case studies.

### 10.2 When to Use This Knowledge
*   **Job Interviews:** Obviously.
*   **Board Presentations:** Answering tough questions from Directors.

### 10.3 Critical Success Factors
1.  **Practice:** Rehearse your answers out loud.
2.  **Honesty:** If you don't know, say "I don't know, but here is how I would find out."

### 10.4 Further Reading
*   **Actuarial Outpost:** Discussion forums on interview questions.
*   **CAS/SOA:** "Professionalism Course" materials.

---

## Appendix

### A. Glossary
*   **STAR:** Situation, Task, Action, Result.
*   **Elevator Pitch:** A 30-second summary.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Inverse Power** | $1 + ax^{-b}$ | Tail Fitting |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

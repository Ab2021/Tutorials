# Phase 2 Review & Interview Drill (Part 2) - Theoretical Deep Dive

## Overview
The final day of Phase 2. We shift focus to the "Big Picture" questions: **Solvency II**, **IFRS 17**, and **Capital Optimization**. These are the questions asked in Senior Actuarial Analyst and Manager interviews.

---

## 1. Solvency II & Capital Questions

### 1.1 "Explain the difference between the Standard Formula and an Internal Model."

*   **Answer:**
    *   **Standard Formula:** A "One Size Fits All" calculation prescribed by EIOPA. It uses standard correlations (e.g., 0.25 between Market and Non-Life) and standard shocks (e.g., 39% Equity crash).
    *   **Internal Model:** A bespoke stochastic model built by the insurer. It uses the company's own data to parameterize distributions and copulas.
    *   **Why use IM?** If your risk profile is unique (e.g., Monoline Credit Insurer) or if you want capital relief from diversification that the SF ignores.

### 1.2 "What is the Matching Adjustment?"

*   **Answer:**
    *   It is a mechanism that allows insurers to use a higher discount rate (and thus hold lower liabilities) if they hold long-term assets that match their long-term liabilities (Annuities).
    *   It prevents "Artificial Volatility" in the balance sheet caused by short-term spread widening.

### 1.3 "How would you optimize the Solvency Ratio?"

*   **Answer:**
    1.  **Numerator (Own Funds):** De-risk the asset portfolio (Sell Equity, Buy Bonds) to reduce the risk margin. Issue Subordinated Debt (Tier 2 Capital).
    2.  **Denominator (SCR):** Buy Reinsurance (XoL) to reduce Underwriting Risk. Use Derivatives to hedge Market Risk. Diversify into new lines of business.

---

## 2. IFRS 17 Questions

### 2.1 "Explain the Contractual Service Margin (CSM) to a CFO."

*   **Answer:**
    *   "The CSM is a 'Profit Bucket'. Instead of booking all the profit on Day 1 when we sell the policy, we put the expected profit into this bucket. Every year, as we provide coverage, we release a spoon of profit from the bucket to the P&L. It ensures our earnings are smooth and reflect the service provided."

### 2.2 "When can you use the Premium Allocation Approach (PAA)?"

*   **Answer:**
    *   You can use PAA if the coverage period is one year or less (e.g., most Auto/Home policies).
    *   You can also use it for longer contracts *if* you can prove that the PAA result is not materially different from the GMM result (LRC Materiality Test).

### 2.3 "What happens if a group of contracts becomes Onerous?"

*   **Answer:**
    *   If $PV(Outflows) > PV(Inflows)$, the CSM cannot be negative.
    *   We must recognize the entire loss *immediately* in the P&L.
    *   We create a "Loss Component" on the balance sheet to track this loss. Future profits first go to reversing this Loss Component before rebuilding the CSM.

---

## 3. Risk Management & Governance Questions

### 3.1 "What is the ORSA?"

*   **Answer:**
    *   **Own Risk and Solvency Assessment.**
    *   It is a continuous process (not just a report) where the Board assesses:
        1.  **Solvency Needs:** Do we have enough capital for the business plan?
        2.  **Stress Testing:** Can we survive a 1-in-200 year pandemic?
        3.  **Link to Strategy:** Does our capital position support our growth targets?

### 3.2 "How do you validate an Expert Judgement?"

*   **Answer:**
    *   **Benchmarking:** Compare the judgement to industry data.
    *   **Sensitivity:** Show how much the result changes if the judgement is wrong.
    *   **Governance:** Ensure the judgement is approved by a committee, not just one person.
    *   **Backtesting:** Track the judgement over time against actual experience.

---

## 4. Modeling Artifacts & Implementation

### 4.1 The "Stress Test" Drill

*   **Interviewer:** "Walk me through how you would stress test our capital position for a 'High Inflation' scenario."
*   **Candidate:**
    1.  **Assets:** Inflation leads to higher interest rates. Bond values drop. (Hit to Own Funds).
    2.  **Liabilities (Non-Life):** Claims cost more to settle. Reserve strengthening required. (Hit to Own Funds).
    3.  **Liabilities (Life):** Expenses rise. (Hit to Own Funds).
    4.  **SCR:** The "Inflation Risk" charge might increase.
    5.  **Management Action:** We would re-price policies and buy inflation-linked bonds.

### 4.2 Python Drill: VaR vs. TVaR

```python
import numpy as np

# Generate Loss Distribution (LogNormal)
losses = np.random.lognormal(10, 1, 100000)

# Calculate VaR 99.5%
var_995 = np.percentile(losses, 99.5)

# Calculate TVaR 99.0% (Expected Loss given Loss > VaR 99)
var_990 = np.percentile(losses, 99.0)
tail_losses = losses[losses > var_990]
tvar_990 = np.mean(tail_losses)

print(f"VaR 99.5%: {var_995:,.0f}")
print(f"TVaR 99.0%: {tvar_990:,.0f}")

# Interview Insight:
# "TVaR is coherent (sub-additive). VaR is not. 
# Regulators prefer TVaR (Swiss Solvency Test), but Solvency II uses VaR."
```

---

## 5. Evaluation & Validation

### 5.1 The "Commercial Awareness" Test

*   **Q:** "How does the current high-interest rate environment affect our Solvency II ratio?"
*   **A:**
    *   **Positive:** Discount rates are higher, so the value of liabilities drops significantly (especially for Life insurers).
    *   **Negative:** The value of fixed-income assets drops.
    *   **Net:** Usually positive for Life insurers (Duration Gap: Liab > Assets). Neutral/Negative for Non-Life.

### 5.2 The "Ethics" Test

*   **Q:** "The CEO asks you to change the mortality assumption to release \$10M of profit for the quarter. What do you do?"
*   **A:**
    *   "I would check if the change is actuarially justified by the data."
    *   "If yes, I proceed and document it."
    *   "If no, I explain to the CEO that I cannot sign off on a biased assumption as it violates the Actuarial Code of Conduct. I would propose alternative legitimate ways to manage capital (e.g., Reinsurance)."

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Confusing Accounting with Solvency**
    *   **Q:** "Does IFRS 17 profit equal Solvency II capital generation?"
    *   **A:** "No. IFRS 17 smooths profit via the CSM. Solvency II is a 'Mark-to-Market' balance sheet. They move differently."

2.  **Trap: The "Black Box"**
    *   **Q:** "Why did the SCR go up?"
    *   **A:** "The model said so." (Fail).
    *   **A:** "The SCR increased because we increased our exposure to Equities, and the correlation between Equity and Spread risk drove the aggregation higher." (Pass).

---

## 7. Advanced Topics & Extensions

### 7.1 Climate Risk

*   **Q:** "How do we model Climate Change in the ORSA?"
*   **A:**
    *   **Physical Risk:** Adjusting Catastrophe models for higher frequency of floods/storms.
    *   **Transition Risk:** Stress testing the asset portfolio for a collapse in "Brown" industries (Oil & Gas).

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Use Test"

*   For an Internal Model to be approved, it must pass the "Use Test".
*   **Meaning:** The model is not just for the regulator. It is used for:
    *   Pricing.
    *   Reinsurance purchase.
    *   Capital allocation.
    *   Strategy.

---

## 9. Practical Example

### 9.1 Worked Example: The "Dividend Decision"

**Scenario:**
*   Solvency Ratio = 180%.
*   Target Ratio = 150%.
*   **CEO:** "Let's pay a special dividend to bring us down to 150%."
*   **Actuary:** "Wait. We have a large renewal coming up in Jan 1. If we pay the dividend now, we might breach the ratio when we write the new business. Let's project the ratio forward 12 months first."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Solvency II** protects the policyholder (Balance Sheet).
2.  **IFRS 17** informs the shareholder (P&L).
3.  **Risk Management** is about culture, not just math.

### 10.2 When to Use This Knowledge
*   **Final Round Interviews:** Where they test your strategic thinking.
*   **The Job:** Every day.

### 10.3 Critical Success Factors
1.  **Confidence:** Believe in your analysis.
2.  **Nuance:** The world is not black and white. Acknowledge uncertainty.
3.  **Ethics:** Never compromise on professional standards.

### 10.4 Further Reading
*   **The Actuary Magazine:** Read the latest issues to know "Hot Topics".
*   **Big 4 Newsletters:** Deloitte/PwC Insurance updates.

---

## Appendix

### A. Glossary
*   **ORSA:** Own Risk and Solvency Assessment.
*   **SFCR:** Solvency and Financial Condition Report.
*   **CSM:** Contractual Service Margin.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Solvency Ratio** | $OwnFunds / SCR$ | KPI |
| **CSM Release** | $CSM \times (Units_{curr} / Units_{tot})$ | IFRS 17 |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

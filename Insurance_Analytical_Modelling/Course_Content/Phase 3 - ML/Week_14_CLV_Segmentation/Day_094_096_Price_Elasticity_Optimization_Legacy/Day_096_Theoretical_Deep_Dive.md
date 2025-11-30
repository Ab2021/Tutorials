# Price Elasticity & Optimization (Part 3) - Dynamic Pricing & Regulatory Ethics - Theoretical Deep Dive

## Overview
"The algorithm says charge more. The law says be fair."
Pricing is the most regulated part of insurance.
While ML can find "Optimal" prices, they might be illegal (Price Discrimination) or unethical (Redlining).
This day focuses on **Dynamic Pricing**, **Fairness**, and the **Regulatory Landscape** of AI pricing.

---

## 1. Conceptual Foundation

### 1.1 Dynamic Pricing

*   **Definition:** Changing prices in real-time based on demand, time, or behavior.
*   **Examples:**
    *   **Telematics:** Price changes monthly based on driving score.
    *   **Travel Insurance:** Price changes based on flight demand.
    *   **Gig Economy:** "Surge Pricing" for delivery insurance.

### 1.2 The Fairness Dilemma

*   **Economic Fairness:** "Price should equal Risk." (Actuarial View).
*   **Social Fairness:** "Price should not penalize protected classes." (Regulatory View).
*   **Conflict:** If a protected class (e.g., Age) is correlated with Risk, Actuarial Fairness violates Social Fairness.

---

## 2. Mathematical Framework

### 2.1 Fairness Metrics

1.  **Demographic Parity:** Average Price should be equal for Group A and Group B. (Usually too strict).
2.  **Equalized Odds:** The Error Rate (Loss Ratio) should be equal for Group A and Group B.
    *   $E[Loss Ratio | A] = E[Loss Ratio | B]$.
3.  **Disparate Impact:** Ratio of acceptance rates. $\frac{P(Accept|A)}{P(Accept|B)} > 0.8$.

### 2.2 Penalized Optimization

*   **Method:** Add a "Fairness Penalty" to the optimization objective.
*   **Objective:** $Maximize \ Profit - \lambda \times |Price_A - Price_B|$.
*   **Result:** A trade-off between Profit and Fairness.

---

## 3. Theoretical Properties

### 3.1 Proxy Variables

*   **Issue:** You remove "Race" from the model.
*   **Reality:** The model uses "Zip Code" or "Credit Score" as a proxy for Race.
*   **Regulation:** Regulators (NY DFS) are now banning "Unintentional Discrimination" via proxies.

### 3.2 Causal Fairness

*   **Concept:** Using Causal Inference (Do-Calculus) to ensure the price is caused by *Risk*, not by *Identity*.
*   **Counterfactual:** "If this customer were a different race, but had the same driving history, would the price change?"

---

## 4. Modeling Artifacts & Implementation

### 4.1 Measuring Disparate Impact (Fairlearn)

```python
from fairlearn.metrics import demographic_parity_difference, selection_rate
import pandas as pd

# Data: y_pred (Price), sensitive_features (Gender)
# Check if High Price (> $500) is distributed equally
y_high_price = (df['price'] > 500).astype(int)

diff = demographic_parity_difference(
    y_true=y_high_price, # Dummy target
    y_pred=y_high_price,
    sensitive_features=df['gender']
)

print(f"Demographic Parity Difference: {diff:.2f}")
# If > 0.1, you might have a regulatory problem.
```

### 4.2 Adversarial Debiasing

*   **Architecture:**
    *   **Predictor:** Predicts Price.
    *   **Adversary:** Tries to predict Gender from the Price.
*   **Training:** Train the Predictor to *fool* the Adversary.
*   **Result:** A Price that contains no information about Gender.

---

## 5. Evaluation & Validation

### 5.1 The "Rate Filing" Packet

*   **Requirement:** When submitting an ML model to a regulator (DOI), you must provide:
    1.  **Variable List:** Every input.
    2.  **Monotonicity Constraints:** "Price must go up as Accidents go up."
    3.  **Impact Analysis:** Effect on protected classes.

### 5.2 Explainability (SHAP)

*   **Requirement:** You must explain *why* the price is \$500.
*   **Tool:** SHAP values.
    *   "Base Rate: \$300. + Acccident: \$100. + Credit: \$50. + Location: \$50."

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Black Box" Ban

*   **Trend:** Some states (California) ban "Complex Models" for Auto pricing.
*   **Constraint:** You must use GLMs (Generalized Linear Models) with multiplicative factors. No Random Forests.
*   **Workaround:** Distill the Random Forest into a GLM (Model Distillation).

### 6.2 Price Optimization Bans

*   **Specific Ban:** "Price Optimization" (charging based on elasticity) is explicitly banned in ~20 states.
*   **Compliance:** You must prove your price factors are based on *Cost*, not *Willingness to Pay*.

---

## 7. Advanced Topics & Extensions

### 7.1 Personalized Bundling

*   **Loophole:** You can't change the *Price* of the base product, but you can create a *Custom Bundle*.
*   **Dynamic:** "For you, we recommend the Gold Plan with a \$500 deductible."

### 7.2 Usage-Based Insurance (UBI)

*   **Fairness:** UBI is considered "Fairer" because it prices based on *Behavior* (How you drive), not *Demographics* (Who you are).
*   **Privacy:** The trade-off is surveillance.

---

## 8. Regulatory & Governance Considerations

### 8.1 GDPR & Automated Decision Making

*   **Article 22:** Right to "Human Intervention".
*   **Impact:** If a customer contests an AI price, a human underwriter must be able to review and override it.

---

## 9. Practical Example

### 9.1 The "Credit Score" Debate

**Scenario:** Credit Score is highly predictive of Risk (Loss Ratio).
**Issue:** Credit Score is highly correlated with Race/Income.
**Action:**
1.  **Regulator:** Bans Credit Score in pricing (e.g., Washington state attempt).
2.  **Insurer:** Must find alternative variables (Telematics, Payment History) to recover the predictive power.
**Result:** Innovation in feature engineering driven by regulation.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Fairness** is mathematical and legal.
2.  **Proxies** are the new battleground.
3.  **Transparency** is non-negotiable.

### 10.2 When to Use This Knowledge
*   **Compliance:** Auditing models before deployment.
*   **Product:** Designing UBI programs.

### 10.3 Critical Success Factors
1.  **Early Legal Review:** Don't build a model for 6 months only to find out the variable is illegal.
2.  **Documentation:** Document every decision. "Why did we include variable X?"

### 10.4 Further Reading
*   **Barocas & Selbst:** "Big Data's Disparate Impact".
*   **NAIC:** "Artificial Intelligence Principles".

---

## Appendix

### A. Glossary
*   **Redlining:** Denying coverage to specific neighborhoods (Illegal).
*   **Disparate Impact:** Unintentional discrimination.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Demographic Parity** | $P(\hat{Y}=1|A) = P(\hat{Y}=1|B)$ | Fairness Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

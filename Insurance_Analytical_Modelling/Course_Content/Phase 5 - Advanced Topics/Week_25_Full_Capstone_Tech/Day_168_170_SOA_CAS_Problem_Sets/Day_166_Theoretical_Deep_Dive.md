# Model Risk Management & Documentation (Part 2) - Automated Documentation & Testing - Theoretical Deep Dive

## Overview
"If it isn't documented, it's a black box. If it isn't tested, it's a ticking time bomb."
Manual documentation is always out of date. Manual testing is always skipped.
This day focuses on **MLOps for Actuaries**: How to use code to document and test itself.
We introduce **Model Cards**, **Great Expectations**, and **Property-Based Testing**.

---

## 1. Conceptual Foundation

### 1.1 The "Documentation Debt" Spiral

1.  Actuary builds model.
2.  Actuary writes 50-page Word doc.
3.  Model changes next week.
4.  Word doc is not updated.
5.  6 months later, the doc describes a model that no longer exists.
*   **Solution:** Documentation as Code. The docs live *in* the repo and are generated *by* the build pipeline.

### 1.2 The Testing Pyramid for ML

*   **Unit Tests:** Does `calculate_age()` handle leap years correctly? (Fast, Specific).
*   **Integration Tests:** Does the data pipeline feed the model correctly?
*   **Model Tests:** Does the model accuracy meet the threshold? (Slow, Stochastic).
*   **Data Tests:** Is the input data valid? (Great Expectations).

---

## 2. Mathematical Framework

### 2.1 Property-Based Testing (Hypothesis)

*   **Traditional Unit Test:** `assert add(2, 2) == 4`.
*   **Property-Based Test:** `for all x, y: assert add(x, y) == add(y, x)`.
*   **Application:**
    *   *Monotonicity:* Increasing deductible should *never* increase premium.
    *   *Bounds:* Predicted probability must be $[0, 1]$.

### 2.2 Model Cards

A standardized "Nutrition Label" for models (Google).
*   **Intended Use:** "Pricing Personal Auto in Texas."
*   **Limitations:** "Not valid for drivers < 16."
*   **Training Data:** "1M policies from 2018-2022."
*   **Performance:** "RMSE = 50. Bias = 0.02."

---

## 3. Theoretical Properties

### 3.1 Deterministic vs. Stochastic Testing

*   **Code is Deterministic:** `2 + 2 = 4` always.
*   **ML is Stochastic:** `train(data)` might produce slightly different weights each time.
*   **Fix:**
    *   Set `random_seed`.
    *   Use "Approximate Assertions" (`assert almost_equal`).
    *   Test the *distribution* of outputs, not single points.

### 3.2 Data Drift Detection

*   **Concept:** The world changes.
*   **Metric:** Population Stability Index (PSI) or Kullback-Leibler (KL) Divergence.
    *   $PSI = \sum (P_{actual} - P_{expected}) \ln(P_{actual} / P_{expected})$.
    *   If $PSI > 0.2$, the model needs retraining.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Automated Model Card Generation

```python
import json

class ModelCard:
    def __init__(self, model_name, author):
        self.metadata = {
            "name": model_name,
            "author": author,
            "metrics": {},
            "parameters": {}
        }
    
    def log_metric(self, key, value):
        self.metadata["metrics"][key] = value
        
    def save(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.metadata, f, indent=4)

# Usage in Pipeline
card = ModelCard("Claims_GLM", "Jane Doe")
card.log_metric("RMSE", 125.4)
card.log_metric("Training_Rows", 50000)
card.save("model_card.json")
```

### 4.2 Property-Based Testing with Hypothesis

```python
from hypothesis import given, strategies as st
import pytest

def calculate_premium(base_rate, age_factor, deductible_factor):
    return base_rate * age_factor * deductible_factor

@given(
    base_rate=st.floats(min_value=100, max_value=1000),
    age_factor=st.floats(min_value=0.5, max_value=3.0),
    deductible_factor=st.floats(min_value=0.1, max_value=1.0)
)
def test_premium_is_positive(base_rate, age_factor, deductible_factor):
    premium = calculate_premium(base_rate, age_factor, deductible_factor)
    assert premium > 0

@given(
    deductible_low=st.floats(min_value=0.8, max_value=0.9),
    deductible_high=st.floats(min_value=0.1, max_value=0.7) # Higher deductible = Lower factor
)
def test_deductible_monotonicity(deductible_low, deductible_high):
    # Logic: Higher deductible (lower factor) should yield lower premium
    p_low = calculate_premium(100, 1.0, deductible_low)
    p_high = calculate_premium(100, 1.0, deductible_high)
    assert p_high < p_low
```

---

## 5. Evaluation & Validation

### 5.1 Great Expectations (Data Validation)

*   **Concept:** Assertions for data.
*   **Examples:**
    *   `expect_column_values_to_be_unique("policy_id")`
    *   `expect_column_values_to_be_between("driver_age", 16, 100)`
    *   `expect_column_mean_to_be_between("claim_severity", 5000, 7000)`
*   **Workflow:** Run this *before* training. If it fails, stop the pipeline.

### 5.2 The "Golden Set" Test

*   **Method:** Keep a static dataset of 100 complex policies with "Known Correct" premiums.
*   **Check:** Every time you update the model, run the Golden Set.
*   **Goal:** Ensure no regression on key business cases.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 "It works on my machine"

*   **Problem:** Dependency hell.
*   **Solution:** Docker. Containerize the training environment.
*   **Documentation:** The `Dockerfile` *is* the documentation of the environment.

### 6.2 Over-Testing

*   **Issue:** Writing tests for `pandas` functions (e.g., testing that `df.mean()` calculates a mean).
*   **Rule:** Test *your* logic, not the library's logic.

---

## 7. Advanced Topics & Extensions

### 7.1 Continuous Integration / Continuous Deployment (CI/CD)

*   **Workflow:**
    1.  Push code to Git.
    2.  GitHub Actions triggers.
    3.  Run Unit Tests.
    4.  Run Great Expectations.
    5.  Train Model on small subset.
    6.  Generate Model Card.
    7.  If all pass, merge.

### 7.2 Shadow Mode

*   **Deployment:** Deploy the new model alongside the old one.
*   **Action:** The old model makes the decisions. The new model just logs its predictions.
*   **Validation:** Compare them for 1 month before switching.

---

## 8. Regulatory & Governance Considerations

### 8.1 Audit Trails

*   **Requirement:** "Show me the code that generated the rates on Jan 1st, 2023."
*   **Solution:** Git Tags + MLflow.
    *   "Tag v1.2.3 corresponds to Model Artifact hash 8a7b..."

---

## 9. Practical Example

### 9.1 The "Silent Failure"

**Scenario:**
*   Data vendor changes "Gender" from "M/F" to "Male/Female".
*   Model expects "M".
*   One-Hot Encoder creates a new column "Gender_Male" and drops "Gender_M".
*   Model silently ignores the new column and uses the intercept.
*   **Result:** Gender factor becomes 1.0 for everyone. Rates are wrong.
**Fix:**
*   **Great Expectations:** `expect_column_distinct_values_to_be_in_set("Gender", ["M", "F"])`.
*   **Result:** Pipeline crashes immediately. Data Scientist is alerted. Crisis averted.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Docs as Code.**
2.  **Test Data, Code, and Models.**
3.  **Model Cards** provide transparency.

### 10.2 When to Use This Knowledge
*   **Setting up a Team:** "We need a CI/CD pipeline before we build the first model."
*   **Refactoring:** Cleaning up legacy spaghetti code.

### 10.3 Critical Success Factors
1.  **Automation:** If it requires a human to click a button, it will be forgotten.
2.  **Culture:** "Tests are not extra work. They are part of the work."

### 10.4 Further Reading
*   **Google:** "Model Cards for Model Reporting".
*   **Great Expectations:** Documentation.

---

## Appendix

### A. Glossary
*   **CI/CD:** Continuous Integration / Continuous Deployment.
*   **Unit Test:** Testing a single function in isolation.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PSI** | $\sum (A-E) \ln(A/E)$ | Drift Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

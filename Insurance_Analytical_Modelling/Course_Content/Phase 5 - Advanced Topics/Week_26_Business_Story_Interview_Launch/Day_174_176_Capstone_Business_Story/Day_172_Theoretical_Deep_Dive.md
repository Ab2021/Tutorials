# Full Capstone Project Build (Part 2) - Modeling & Validation - Theoretical Deep Dive

## Overview
"A model that predicts 0 loss for everyone has a low RMSE, but it's useless."
In insurance, standard metrics like MSE or Accuracy don't work well.
We need **Lift**, **Gini**, and **Double Lift**.
This day focuses on training the **Pricing Engine** (XGBoost/LightGBM) and proving it beats the legacy GLM.

---

## 1. Conceptual Foundation

### 1.1 The GLM vs. GBM Debate

*   **GLM (Generalized Linear Model):**
    *   *Pros:* Transparent, additive, easy to explain to regulators.
    *   *Cons:* Misses non-linear interactions (e.g., Young Driver + High Power Car).
*   **GBM (Gradient Boosting Machine):**
    *   *Pros:* Captures complex interactions automatically. High predictive power.
    *   *Cons:* "Black Box" (requires SHAP values to explain).
*   **Capstone Strategy:** Train *both*. Use the GLM as the baseline to beat.

### 1.2 The Tweedie Distribution

*   **Why:** Insurance data has a "Point Mass at Zero" (Most people don't crash).
*   **Tweedie:** A compound distribution (Poisson Frequency + Gamma Severity).
*   **Variance Power ($p$):**
    *   $p=1$: Poisson.
    *   $p=2$: Gamma.
    *   $1 < p < 2$: Compound Poisson-Gamma (Insurance Sweet Spot).

---

## 2. Mathematical Framework

### 2.1 The Gini Coefficient (Insurance Version)

*   **Lorenz Curve:** Plot Cumulative Exposure vs. Cumulative Loss (sorted by Predicted Loss Cost).
*   **Perfect Model:** Sorts all losses to the right.
*   **Random Model:** Diagonal line.
*   **Gini:** $2 \times \text{Area between Curve and Diagonal}$.
*   **Target:** A Gini of 0.30 - 0.40 is excellent for Auto Insurance.

### 2.2 Double Lift Chart

*   **Goal:** Compare Model A (New) vs. Model B (Old).
*   **Method:**
    1.  Sort data by the ratio `Pred_A / Pred_B`.
    2.  Bin into deciles.
    3.  Calculate Actual Loss Ratio for each bin.
*   **Interpretation:** If the slope is positive, Model A is better. It correctly identifies that the policies it thinks are expensive (relative to B) actually *are* expensive.

---

## 3. Theoretical Properties

### 3.1 Offset / Base Margin

*   **Concept:** We are modeling the *Rate*, not the absolute loss.
*   **Implementation:**
    *   GLM: `offset = log(exposure)`.
    *   XGBoost: `base_margin = log(exposure)`.
*   **Result:** The model predicts `Loss / Exposure` (Pure Premium).

### 3.2 Monotonicity Constraints

*   **Regulation:** You cannot charge a 40-year-old *more* than a 20-year-old (all else equal).
*   **Constraint:** Force the "Age" curve to be monotonic decreasing.
*   **XGBoost:** `monotone_constraints`.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Training the XGBoost Model

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Prepare Data
X = df[["vehicle_age", "driver_age", "prior_claims", "credit_score"]]
y = df["incurred_loss"]
exposure = df["exposure"]

# 2. Split (Time-Based)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
exp_train, exp_test = exposure[:len(X_train)], exposure[len(X_train):]

# 3. DMatrix (Optimized Data Structure)
dtrain = xgb.DMatrix(X_train, label=y_train)
dtrain.set_base_margin(np.log(exp_train)) # Crucial!

dtest = xgb.DMatrix(X_test, label=y_test)
dtest.set_base_margin(np.log(exp_test))

# 4. Parameters
params = {
    "objective": "reg:tweedie",
    "tweedie_variance_power": 1.5, # Standard for insurance
    "learning_rate": 0.1,
    "max_depth": 4,
    "monotone_constraints": "(0, -1, 1, 0)" # Example constraints
}

# 5. Train
model = xgb.train(params, dtrain, num_boost_round=1000, 
                  evals=[(dtest, "Test")], early_stopping_rounds=50)

# 6. Predict
preds = model.predict(dtest) # Returns Total Loss (because of base_margin)
pure_premium = preds / exp_test
```

### 4.2 Calculating Gini Coefficient

```python
def gini(actual, pred, weight):
    # Sort by Prediction
    df_sorted = pd.DataFrame({"actual": actual, "pred": pred, "weight": weight})
    df_sorted = df_sorted.sort_values("pred")
    
    # Cumulative Sums
    cum_weight = df_sorted["weight"].cumsum() / df_sorted["weight"].sum()
    cum_actual = df_sorted["actual"].cumsum() / df_sorted["actual"].sum()
    
    # Area under Lorenz Curve
    # Trapezoidal Rule
    area = np.trapz(cum_actual, cum_weight)
    
    # Gini = 2 * (0.5 - Area) (Note: Definition varies slightly, check orientation)
    gini_score = 1 - 2 * area
    return gini_score

print(f"Gini Score: {gini(y_test, preds, exp_test):.4f}")
```

---

## 5. Evaluation & Validation

### 5.1 The "Lift Chart"

*   **X-Axis:** Deciles of Predicted Pure Premium.
*   **Y-Axis:** Average Actual Pure Premium.
*   **Visual:** Should be a steep upward slope.
    *   Decile 1 (Safest): Predicted \$100, Actual \$105.
    *   Decile 10 (Riskiest): Predicted \$2000, Actual \$1950.
*   **Validation:** If the line is flat, the model is random guessing.

### 5.2 Stability Check

*   **Test:** Run the model on 2021 data and 2022 data.
*   **Check:** Do the feature importances change? Does the Gini drop significantly?
*   **Overfitting:** If Train Gini = 0.60 and Test Gini = 0.20, you overfit.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Zero Exposure" Bug

*   **Issue:** `log(exposure)` where exposure is 0 (or very small).
*   **Result:** `-inf` or massive negative numbers.
*   **Fix:** `np.log(np.maximum(exposure, 0.001))`.

### 6.2 Correlation vs. Causation

*   **Feature:** "Red Car".
*   **Model:** Predicts higher risk.
*   **Reality:** Red cars are often Sports Cars. The *car type* causes the risk, not the paint.
*   **Action:** Include "Vehicle Type" in the model to control for this.

---

## 7. Advanced Topics & Extensions

### 7.1 SHAP Values (Explainability)

*   **Requirement:** You must tell the regulator *why* you increased someone's rate.
*   **Tool:** `shap.TreeExplainer(model)`.
*   **Output:** "Your rate is high because (1) You are 18 (+50%), (2) You drive a Mustang (+30%)."

### 7.2 Generalized Additive Models (GAMs)

*   **Middle Ground:** `pygam`.
*   **Structure:** Linear combination of non-linear splines.
*   **Benefit:** Visualization. You can plot the exact curve for "Age".

---

## 8. Regulatory & Governance Considerations

### 8.1 Disparate Impact Testing

*   **Check:** Does the model unintentionally discriminate against protected classes?
*   **Method:** Calculate "Average Prediction" by Race/Gender (using proxy data if necessary).
*   **Rule:** If the ratio is outside 0.8 - 1.25, you have a problem.

---

## 9. Practical Example

### 9.1 The "Credit Score" Ban

**Scenario:** Washington state bans the use of Credit Score in insurance.
**Impact:** Your model relies heavily on it.
**Action:**
1.  Retrain model without `credit_score`.
2.  Gini drops from 0.35 to 0.28.
3.  **Business Decision:** We must find a substitute (e.g., Telematics) or accept lower profitability.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Tweedie** is the standard loss function.
2.  **Gini** measures segmentation power.
3.  **Lift Charts** prove profitability.

### 10.2 When to Use This Knowledge
*   **Capstone:** This is the core modeling step.
*   **Interviews:** "How do you evaluate an insurance model?" (Answer: Lift Charts, not RMSE).

### 10.3 Critical Success Factors
1.  **Base Margin:** Never forget the exposure offset.
2.  **Hyperparameters:** Don't just use defaults. Tune `tweedie_variance_power`.

### 10.4 Further Reading
*   **XGBoost Docs:** "Tweedie Regression".
*   **Frees, Derrig, Meyers:** "Predictive Modeling Applications in Actuarial Science".

---

## Appendix

### A. Glossary
*   **Pure Premium:** Loss / Exposure.
*   **Relativity:** The multiplier for a specific factor level (e.g., Age 20 relativity = 1.50).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Tweedie Variance** | $Var(Y) = \phi \mu^p$ | Loss Distribution |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

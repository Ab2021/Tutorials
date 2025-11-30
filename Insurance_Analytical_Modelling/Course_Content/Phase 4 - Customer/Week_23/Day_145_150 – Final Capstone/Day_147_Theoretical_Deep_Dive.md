# Final Capstone: Modeling & Experimentation (Part 3) - MLflow & Tuning - Theoretical Deep Dive

## Overview
"Science is the difference between 'It works' and 'I know why it works'."
In the Capstone, you are not just training *one* model. You are running *experiments*.
You need to track every Hyperparameter, every Metric, and every Artifact.

---

## 1. Conceptual Foundation

### 1.1 The Experiment Lifecycle

1.  **Baseline:** Logistic Regression (The "Control Group").
2.  **Challenger 1:** Random Forest (Non-linear).
3.  **Challenger 2:** XGBoost (Gradient Boosting).
4.  **Challenger 3:** Neural Network (Deep Learning).
5.  **Selection:** Best ROI (not just best Accuracy).

### 1.2 The "Black Box" Problem

*   **Issue:** You train a model today. It works. Six months later, you try to retrain it, but you forgot the `random_state` or the `max_depth`.
*   **Solution:** **MLflow Tracking**.
    *   *Logs:* Parameters, Metrics, Code Version (Git Commit), Model Artifact (.pkl).

---

## 2. Mathematical Framework

### 2.1 Hyperparameter Tuning Strategies

1.  **Grid Search:** Brute force. Try *every* combination.
    *   *Cost:* $O(N^k)$. (Exponential).
2.  **Random Search:** Try random combinations.
    *   *Theory:* High-dimensional spaces are mostly empty. Random search finds the "good enough" region faster.
3.  **Bayesian Optimization (Optuna):**
    *   *Concept:* Build a probability model $P(\text{Score} | \text{Params})$.
    *   *Action:* Pick the next set of params that maximizes the **Expected Improvement (EI)**.

### 2.2 Cross-Validation (Time Series)

*   **Standard CV:** K-Fold. (Bad for Insurance).
    *   *Why:* Leakage. You can't use 2024 data to predict 2023.
*   **Time Series Split:**
    *   *Fold 1:* Train (Jan-Mar), Test (Apr).
    *   *Fold 2:* Train (Jan-Apr), Test (May).
    *   *Fold 3:* Train (Jan-May), Test (Jun).

---

## 3. Theoretical Properties

### 3.1 The Model Registry

*   **Stages:**
    *   **None:** Just an experiment.
    *   **Staging:** Passed unit tests. Ready for QA.
    *   **Production:** The live model serving traffic.
    *   **Archived:** The old model (kept for audit).
*   **Governance:** You cannot promote to Production without a "Sign-off" (Manual or Automated).

### 3.2 Calibration

*   **Problem:** XGBoost outputs a score (0.8), not a probability.
*   **Requirement:** For Insurance Pricing, you need *actual probabilities*.
    *   If Model says 10% risk, then 10% of those people *must* claim.
*   **Fix:** **Isotonic Regression** or **Platt Scaling**.

---

## 4. Modeling Artifacts & Implementation

### 4.1 MLflow Training Loop (Python)

```python
import mlflow
import xgboost as xgb
from sklearn.metrics import roc_auc_score

# 1. Start Run
with mlflow.start_run(run_name="XGB_Experiment_1"):
    
    # 2. Define Params
    params = {
        "max_depth": 5,
        "learning_rate": 0.1,
        "n_estimators": 100
    }
    
    # 3. Log Params
    mlflow.log_params(params)
    
    # 4. Train
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    # 5. Log Metrics
    y_pred = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred)
    mlflow.log_metric("auc", auc)
    
    # 6. Log Model
    mlflow.xgboost.log_model(model, "model")
    
    print(f"Run Complete. AUC: {auc}")
```

### 4.2 Optuna Optimization

```python
import optuna

def objective(trial):
    # 1. Suggest Params
    param = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
    }
    
    # 2. Train & Eval
    model = xgb.XGBClassifier(**param)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    
    return score

# 3. Optimize
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=100)
```

---

## 5. Evaluation & Validation

### 5.1 The "Lift Chart" (Business Metric)

*   **X-Axis:** Deciles of Risk (Top 10% riskiest to Bottom 10%).
*   **Y-Axis:** Actual Loss Ratio.
*   **Goal:** A steep slope. The Top 10% should have a *much* higher Loss Ratio than the Bottom 10%.
*   **Validation:** If the slope is flat, the model is useless for pricing.

### 5.2 SHAP Values (Explainability)

*   **Requirement:** "Why did you deny my claim?"
*   **Answer:** SHAP Force Plot.
    *   *Base Value:* 10% Risk.
    *   *Feature 1 (Age=21):* +5%.
    *   *Feature 2 (Speed=90):* +20%.
    *   *Final Prediction:* 35% Risk.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Target Leakage via ID**
    *   *Scenario:* The `Policy_ID` contains the year. `POL-2023-123`.
    *   *Result:* The model learns that "2023" policies haven't claimed yet (because they are new).
    *   *Fix:* Drop IDs.

2.  **Trap: Metric Hacking**
    *   *Scenario:* Optimizing for Accuracy on an Imbalanced Dataset (99% Non-Fraud).
    *   *Result:* Model predicts "No Fraud" always. Accuracy = 99%. Recall = 0%.
    *   *Fix:* Use PR-AUC (Precision-Recall Area Under Curve).

---

## 7. Advanced Topics & Extensions

### 7.1 Stacking (Ensemble Learning)

*   **Concept:** Train a "Meta-Model" (Logistic Regression) that takes the predictions of XGBoost, Random Forest, and Neural Net as inputs.
*   **Benefit:** Usually squeezes out the last 1-2% of performance.

### 7.2 AutoML (H2O / DataRobot)

*   **Role in Capstone:** Use it as a *Benchmark*.
*   **Logic:** If your hand-tuned model can't beat the AutoML baseline, you are doing something wrong.

---

## 8. Regulatory & Governance Considerations

### 8.1 Reproducibility Audit

*   **Test:** Can a colleague clone your repo, run `main.py`, and get the *exact same* AUC?
*   **Requirement:** `requirements.txt`, `seed` setting, and Docker.

---

## 9. Practical Example

### 9.1 Worked Example: The "Model Bake-off"

**Goal:** Select the best Churn Model.
**Experiments:**
1.  **LogReg:** AUC 0.65. Interpretable. Fast.
2.  **XGBoost:** AUC 0.82. Hard to explain. Slow.
3.  **AutoML:** AUC 0.83. Black box.
**Decision:**
*   Choose **XGBoost**.
*   Why? The lift (0.82 vs 0.65) is worth the complexity cost.
*   Mitigation: Use SHAP for explainability.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Track Everything (MLflow).**
2.  **Tune Intelligently (Optuna).**
3.  **Validate for Business (Lift Charts).**

### 10.2 When to Use This Knowledge
*   **Capstone Presentation:** Show the MLflow Dashboard screenshot. "I ran 50 experiments."
*   **Code Review:** "Where is the experiment tracking?"

### 10.3 Critical Success Factors
1.  **Discipline:** Don't change code without committing to Git.
2.  **Patience:** Hyperparameter tuning takes time. Run it overnight.

### 10.4 Further Reading
*   **Databricks:** "The MLflow Guide".

---

## Appendix

### A. Glossary
*   **AUC:** Area Under the Curve.
*   **SHAP:** SHapley Additive exPlanations.
*   **CV:** Cross-Validation.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Log Loss** | $-\frac{1}{N} \sum (y \log p + (1-y) \log (1-p))$ | Classification Error |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

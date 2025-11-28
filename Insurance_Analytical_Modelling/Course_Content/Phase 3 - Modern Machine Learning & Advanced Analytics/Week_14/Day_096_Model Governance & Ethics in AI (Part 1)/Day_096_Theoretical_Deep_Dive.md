# MLOps for Actuaries (Part 1) - Theoretical Deep Dive

## Overview
"Which version of the model is in production? The one on Bob's laptop or the one in the shared folder?" This session covers **MLOps (Machine Learning Operations)**, specifically **Experiment Tracking** with **MLflow** to escape "Excel Hell".

---

## 1. Conceptual Foundation

### 1.1 The Reproducibility Crisis

*   **Scenario:** A regulator asks you to reproduce the pricing model from 2022.
*   **Problem:** You have the code, but you lost the specific *hyperparameters* and the exact *training data snapshot*.
*   **Solution:** Experiment Tracking. Every time you press "Run", you log everything.

### 1.2 What to Track?

1.  **Parameters:** `learning_rate`, `max_depth`, `subsample`.
2.  **Metrics:** `RMSE`, `Gini`, `Lift_Top_Decile`.
3.  **Artifacts:** The model file (`model.pkl`), the feature importance plot (`importance.png`), the data schema.
4.  **Source:** The Git Commit Hash (exact version of the code).

### 1.3 MLflow

*   The industry standard (Open Source, Databricks).
*   **Components:**
    *   **Tracking:** Logging runs.
    *   **Projects:** Packaging code.
    *   **Models:** Packaging models.
    *   **Registry:** Managing lifecycle (Staging -> Production).

---

## 2. Mathematical Framework

### 2.1 Hyperparameter Tuning (Grid vs. Random vs. Bayesian)

*   **Grid Search:** Try *every* combination. (Slow).
*   **Random Search:** Try random combinations. (Faster, often better).
*   **Bayesian Optimization (Hyperopt):** Learn from past runs. "Run 1 had `depth=3` and was bad. Run 2 had `depth=10` and was good. Try `depth=9` next."

### 2.2 The Metric Log

$$ \text{Run}_i = \{ \theta_i, M(\theta_i) \} $$
*   We want to find $\theta^*$ such that $M(\theta^*)$ is minimized.
*   MLflow visualizes $M(\theta)$ vs. $\theta$ (Parallel Coordinates Plot).

---

## 3. Theoretical Properties

### 3.1 Idempotency

*   Running the same code twice should produce the same result.
*   *Requirement:* Fix the Random Seed (`random_state=42`).

### 3.2 Model Lineage

*   **Data Lineage:** Where did the data come from? (SQL Query).
*   **Code Lineage:** Which Git Commit?
*   **Model Lineage:** Which Run produced this `.pkl` file?

---

## 4. Modeling Artifacts & Implementation

### 4.1 MLflow Tracking (Python)

```python
import mlflow
import mlflow.xgboost
from sklearn.metrics import mean_squared_error

# 1. Set Experiment
mlflow.set_experiment("Auto_Pricing_v1")

# 2. Start Run
with mlflow.start_run():
    # Define Params
    params = {
        "max_depth": 6,
        "learning_rate": 0.1,
        "n_estimators": 100
    }
    
    # Log Params
    mlflow.log_params(params)
    
    # Train Model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    
    # Evaluate
    predictions = model.predict(X_test)
    rmse = mean_squared_error(y_test, predictions, squared=False)
    
    # Log Metrics
    mlflow.log_metric("rmse", rmse)
    
    # Log Model (Artifact)
    mlflow.xgboost.log_model(model, "model")
    
    print(f"Run ID: {mlflow.active_run().info.run_id}")
```

### 4.2 Comparing Runs

*   **UI:** Go to `http://localhost:5000`.
*   **Table:** Sort by `rmse` ascending.
*   **Parallel Coordinates:** Visualize which params drive performance.

---

## 5. Evaluation & Validation

### 5.1 The "Best" Model

*   Is the model with the lowest RMSE the best?
*   *No.* It might be overfitted.
*   *Actuarial Check:* Look at the Gap between Train RMSE and Test RMSE. If Gap is large, reject.

### 5.2 Artifact Validation

*   Did we log the **SHAP plots**?
*   Did we log the **Lift Chart**?
*   *Rule:* If it's not in MLflow, it didn't happen.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Logging Too Much**
    *   Logging every single row of the training data.
    *   *Fix:* Log a *sample* or a *hash* (checksum) of the data.

2.  **Trap: "Local" MLflow**
    *   Running MLflow on your laptop's `C:` drive.
    *   *Fix:* Use a shared Tracking Server (AWS S3 + RDS) so the whole team can see it.

### 6.2 Implementation Challenges

1.  **Environment Hell:**
    *   "It works on my machine."
    *   *Fix:* MLflow logs the `conda.yaml` file automatically. Use it to recreate the environment.

---

## 7. Advanced Topics & Extensions

### 7.1 Auto-Logging

*   `mlflow.autolog()`
*   Automatically logs params, metrics, and models for supported libraries (Scikit-Learn, XGBoost, TensorFlow) without writing `log_param`.

### 7.2 Nested Runs

*   Used for Hyperparameter Tuning.
*   **Parent Run:** "Tuning Session 1".
*   **Child Runs:** The 50 individual trials.

---

## 8. Regulatory & Governance Considerations

### 8.1 Audit Trails

*   **Regulator:** "Show me the development history of this model."
*   **Actuary:** "Here is the MLflow export showing all 200 experiments, the failed ones, the successful ones, and why we picked this one."

---

## 9. Practical Example

### 9.1 Worked Example: The "Grid Search"

**Scenario:**
*   Tuning an XGBoost model for Frequency.
*   **Grid:**
    *   `max_depth`: [3, 5, 7]
    *   `learning_rate`: [0.01, 0.1]
*   **Execution:**
    *   Script loops through 6 combinations.
    *   Logs 6 runs to MLflow.
*   **Analysis:**
    *   Run 3 (`depth=5`, `lr=0.1`) has best Test Gini.
    *   Run 5 (`depth=7`) is overfitted.
*   **Decision:** Promote Run 3 to Staging.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Track Everything:** Params, Metrics, Artifacts, Code.
2.  **MLflow** is the tool of choice.
3.  **Reproducibility** is the goal.

### 10.2 When to Use This Knowledge
*   **Model Development:** Daily.
*   **Peer Review:** "Send me the Run ID, I'll check your results."

### 10.3 Critical Success Factors
1.  **Discipline:** Never train a model without `start_run()`.
2.  **Naming:** Name your experiments logically (`Project_Date_Description`).

### 10.4 Further Reading
*   **Zaharia et al.:** "Accelerating the Machine Learning Lifecycle with MLflow".

---

## Appendix

### A. Glossary
*   **Artifact:** Any file output by the run (Image, Model, CSV).
*   **Run ID:** Unique hash (e.g., `a1b2c3d4`).
*   **Experiment:** A group of runs.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **RMSE** | $\sqrt{\frac{1}{n}\sum(y-\hat{y})^2}$ | Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

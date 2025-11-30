# MLOps for Actuaries (Part 2) - Theoretical Deep Dive

## Overview
A model on your laptop is a prototype. A model in the **Registry** is a product. This session covers **Model Governance** (Staging vs. Production) and **Deployment** (Docker, APIs) to turn your Python script into a Rating Engine.

---

## 1. Conceptual Foundation

### 1.1 The Model Registry

*   **Problem:** "I updated the model, but the IT team deployed the old one."
*   **Solution:** A centralized database of approved models.
*   **Stages:**
    *   **None:** Experimental.
    *   **Staging:** Ready for testing (UAT).
    *   **Production:** Live.
    *   **Archived:** Retired.

### 1.2 Deployment Patterns

1.  **Batch Scoring (Offline):**
    *   Run the model every night on the entire portfolio.
    *   *Use Case:* Renewal Book repricing, Reserving.
2.  **Real-Time Scoring (Online):**
    *   Model sits behind an API (Application Programming Interface).
    *   *Use Case:* Quoting a new customer on the website (Millisecond latency).

### 1.3 Docker (Containerization)

*   **Problem:** "It works on my machine" (because I have `pandas 1.3.5` and you have `pandas 2.0`).
*   **Solution:** Ship the computer with the code.
*   **Container:** A lightweight, standalone package that includes Code, Runtime, Libraries, and Settings.

---

## 2. Mathematical Framework

### 2.1 Latency Budget

$$ T_{total} = T_{network} + T_{preprocessing} + T_{inference} $$
*   **Requirement:** Total time < 200ms for a website quote.
*   **Inference:** XGBoost is fast (<10ms). Neural Networks can be slow (>100ms).
*   *Optimization:* Convert models to ONNX (Open Neural Network Exchange) format for speed.

### 2.2 Drift Detection

$$ D(P_{train}, P_{prod}) > \epsilon $$
*   If the distribution of production data ($P_{prod}$) drifts too far from training data ($P_{train}$), trigger an alert.
*   *Metric:* PSI (Population Stability Index).

---

## 3. Theoretical Properties

### 3.1 Immutable Artifacts

*   Once a model is in the Registry, it is **Immutable** (Read-Only).
*   You cannot "tweak" a Production model. You must train a *new* version and promote it.

### 3.2 Blue/Green Deployment

*   **Blue:** Current Production Model (Version 1).
*   **Green:** New Model (Version 2).
*   **Switch:** Route 1% of traffic to Green. If no errors, route 100%.
*   *Benefit:* Zero downtime.

---

## 4. Modeling Artifacts & Implementation

### 4.1 MLflow Model Registry

```python
import mlflow.xgboost
from mlflow.tracking import MlflowClient

# 1. Register Model
model_uri = "runs:/d160.../model"
mv = mlflow.register_model(model_uri, "Auto_Pricing_Model")

# 2. Transition to Staging
client = MlflowClient()
client.transition_model_version_stage(
    name="Auto_Pricing_Model",
    version=mv.version,
    stage="Staging"
)

# 3. Load Production Model (for Scoring)
model = mlflow.pyfunc.load_model("models:/Auto_Pricing_Model/Production")
prediction = model.predict(data)
```

### 4.2 Dockerfile for API

```dockerfile
# 1. Base Image (Python)
FROM python:3.9-slim

# 2. Install Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# 3. Copy Code & Model
COPY app.py .
COPY model.pkl .

# 4. Run API (FastAPI)
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
```

---

## 5. Evaluation & Validation

### 5.1 Shadow Mode

*   Deploy the new model alongside the old one.
*   The new model receives inputs and makes predictions, but **does not affect the customer**.
*   *Goal:* Compare New vs. Old predictions in the real world without risk.

### 5.2 Load Testing

*   Can the API handle 1,000 quotes per second?
*   *Tool:* `Locust` (Python load testing library).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Pickle" Trap**
    *   Saving a model with `pickle` is dangerous (security risk, version incompatibility).
    *   *Fix:* Use `mlflow` format or `ONNX`.

2.  **Trap: Feature Skew**
    *   Training data: `Age` is calculated from DOB.
    *   Production data: `Age` is user-input.
    *   *Result:* Mismatch. Logic must be identical.

### 6.2 Implementation Challenges

1.  **Dependency Hell:**
    *   The model needs `scikit-learn 0.24`, but the server has `0.23`.
    *   *Fix:* Docker solves this completely.

---

## 7. Advanced Topics & Extensions

### 7.1 Kubernetes (K8s)

*   Orchestrates Docker containers.
*   If traffic spikes, K8s automatically spins up 10 more copies of your Pricing Model container.

### 7.2 Feature Store

*   A centralized database of features (e.g., "Customer Lifetime Value").
*   Ensures Training and Production use the exact same feature definition.

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Kill Switch"

*   **Requirement:** Ability to instantly revert to the previous model if a bug is found.
*   **MLflow:** `transition_model_version_stage(..., stage="Production")` (Promote old version back).

---

## 9. Practical Example

### 9.1 Worked Example: The "Friday Night" Deploy

**Scenario:**
*   Actuary updates the GLM.
*   **Registry:** Pushes to "Staging".
*   **CI/CD:** Automated tests run (Check accuracy, check bias). **Pass.**
*   **Approval:** Chief Actuary clicks "Approve" in MLflow UI.
*   **Deploy:** Kubernetes updates the API.
*   **Result:** New rates are live on the website in 5 minutes.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Registry** is the single source of truth.
2.  **Docker** ensures reproducibility in production.
3.  **Real-time** requires robust engineering (APIs).

### 10.2 When to Use This Knowledge
*   **IT Collaboration:** Speaking the language of DevOps.
*   **Production:** When your model graduates from Excel.

### 10.3 Critical Success Factors
1.  **Version Control:** Code (Git) + Data (DVC) + Model (MLflow).
2.  **Monitoring:** Watch the model after deployment (Day 98).

### 10.4 Further Reading
*   **Burkov:** "Machine Learning Engineering".

---

## Appendix

### A. Glossary
*   **API:** Application Programming Interface.
*   **CI/CD:** Continuous Integration / Continuous Deployment.
*   **Latency:** Time taken to get a response.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PSI** | $\sum (P_a - P_e) \ln(P_a / P_e)$ | Drift Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

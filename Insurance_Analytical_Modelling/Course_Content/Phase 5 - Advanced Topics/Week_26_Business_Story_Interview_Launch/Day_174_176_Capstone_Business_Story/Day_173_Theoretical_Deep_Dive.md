# Full Capstone Project Build (Part 3) - Deployment & MLOps - Theoretical Deep Dive

## Overview
"A model is only as good as its ability to serve predictions in production."
In Parts 1 & 2, we built and validated the model. Now, we must operationalize it.
Day 173 focuses on **MLOps**, **Containerization**, and **Real-Time Serving** for the Insurance Capstone.
We will transform our Jupyter Notebook into a production-grade API, containerize it with Docker, and discuss scaling with Kubernetes.

---

## 1. Conceptual Foundation

### 1.1 The MLOps Lifecycle in Insurance

*   **Development:** Experimentation (Notebooks).
*   **Continuous Integration (CI):** Testing code quality and unit tests.
*   **Continuous Deployment (CD):** Automating deployment to staging/prod.
*   **Continuous Training (CT):** Retraining pipelines triggered by data drift.
*   **Monitoring:** Tracking accuracy and fairness in real-time.

### 1.2 The "Serving" Pattern

*   **Batch Serving:** Run overnight for renewal book. (High throughput, high latency).
*   **Real-Time Serving:** Run on-demand for new quotes. (Low latency < 200ms).
*   **Hybrid:** Pre-compute complex features (Batch) and serve predictions (Real-Time).

---

## 2. Mathematical Framework

### 2.1 Drift Detection Metrics

*   **Data Drift:** $P(X_{train}) \neq P(X_{prod})$.
    *   **Metric:** Kullback-Leibler (KL) Divergence or Population Stability Index (PSI).
    *   $$ PSI = \sum (P_{actual} - P_{expected}) \times \ln(\frac{P_{actual}}{P_{expected}}) $$
    *   **Rule of Thumb:** PSI > 0.2 indicates significant drift.
*   **Concept Drift:** $P(Y|X_{train}) \neq P(Y|X_{prod})$.
    *   The relationship between risk factors and loss has changed (e.g., post-COVID driving).

### 2.2 Latency Constraints

*   **Little's Law:** $L = \lambda W$.
    *   $L$: Average number of requests in system.
    *   $\lambda$: Arrival rate (Requests/sec).
    *   $W$: Average wait time (Latency).
*   **Goal:** Minimize $W$ to handle high $\lambda$ during peak quoting hours.

---

## 3. Theoretical Properties

### 3.1 Containerization (Docker)

*   **Concept:** "Build once, run anywhere."
*   **Isolation:** Dependencies (Python 3.9, Scikit-Learn 1.0) are locked inside the container.
*   **Reproducibility:** Eliminates "It works on my machine" bugs.

### 3.2 Orchestration (Kubernetes)

*   **Pod:** The smallest unit (one container).
*   **Service:** A stable IP address for a set of pods.
*   **Horizontal Pod Autoscaling (HPA):**
    *   If CPU > 80%, spin up more pods.
    *   If CPU < 20%, kill pods.
    *   **Insurance Use Case:** Scale up during "Open Enrollment" or after a TV ad spot.

---

## 4. Modeling Artifacts & Implementation

### 4.1 FastAPI Serving Application (`app.py`)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI(title="Insurance Capstone API")

# Load Model
model = joblib.load("artifacts/xgb_model.pkl")

class QuoteRequest(BaseModel):
    age: int
    vehicle_value: float
    credit_score: int
    # ... other features

@app.post("/predict_premium")
def predict(request: QuoteRequest):
    data = pd.DataFrame([request.dict()])
    
    # Preprocessing (must match training!)
    # ...
    
    prediction = model.predict(data)[0]
    return {"predicted_premium": float(prediction)}

@app.get("/health")
def health():
    return {"status": "ok"}
```

### 4.2 Dockerfile

```dockerfile
# Base Image
FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code
COPY . .

# Command
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "80"]
```

---

## 5. Evaluation & Validation

### 5.1 Load Testing (Locust)

*   **Scenario:** Simulate 1000 concurrent agents requesting quotes.
*   **Metric:** P99 Latency (99% of requests must be faster than X).
*   **Failure Mode:** If P99 > 1s, the frontend will timeout and we lose the customer.

### 5.2 Shadow Deployment

*   **Method:** Deploy the new model alongside the old one.
*   **Traffic:** 100% to Old (for decision), Copy of 100% to New (for logging).
*   **Goal:** Verify New Model doesn't crash or produce crazy values before letting it affect customers.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Skewed Training-Serving Data

*   **Issue:** Training data was cleaned CSVs. Serving data is raw JSON.
*   **Bug:** "NaN" in CSV is `np.nan`. "null" in JSON might be `None` or missing key.
*   **Fix:** Use **Pydantic** for strict schema validation at the API gate.

### 6.2 The "Feature Store" Gap

*   **Issue:** Model needs "Average Claims in Zip Code".
*   **Training:** Calculated from full history.
*   **Serving:** How do we get this in < 50ms?
*   **Fix:** **Redis** (Online Feature Store). Pre-compute the value and look it up by Zip Key.

---

## 7. Advanced Topics & Extensions

### 7.1 Model Explainability Service

*   **Requirement:** API must return *why* the price is high.
*   **Implementation:** Run SHAP TreeExplainer inside the API.
*   **Response:** `{"premium": 1200, "reasons": ["Young Driver (+300)", "Sports Car (+200)"]}`.

### 7.2 A/B Testing Infrastructure

*   **Header-Based Routing:**
    *   Request Header `X-Model-Version: v2`.
    *   Load Balancer (Nginx/Istio) routes to the V2 Pods.

---

## 8. Regulatory & Governance Considerations

### 8.1 Audit Trails

*   **Requirement:** Every prediction must be logged.
*   **Log Payload:** Input Vector + Model Version + Output + Timestamp.
*   **Storage:** Write to S3/BigQuery (Async) for compliance auditing.

### 8.2 Fallback Logic

*   **Scenario:** Model Service crashes.
*   **Rule:** Fail Safe.
    *   Return a "Default Rate" (Manual Rating).
    *   Or "Refer to Underwriter".
    *   Never return "Error 500" to the customer.

---

## 9. Practical Example

### 9.1 The "Friday Night" Deploy

**Scenario:** Team deploys a new pricing model on Friday at 4 PM.
**Incident:** The model has a bug where it treats "0 accidents" as "Missing Data".
**Impact:** Quotes for safe drivers skyrocket. Conversion drops to 0%.
**Recovery:**
1.  **Monitoring:** Prometheus alert fires "Conversion Rate Drop".
2.  **Rollback:** Kubernetes command `kubectl rollout undo deployment/pricing-api`.
3.  **Time:** 2 minutes. (Without K8s, might take hours).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **FastAPI** for serving.
2.  **Docker** for packaging.
3.  **Drift Monitoring** for sustainability.

### 10.2 When to Use This Knowledge
*   **Capstone:** The final deliverable is a working API, not just a notebook.
*   **Interview:** "How do you deploy your models?" is a standard Senior Data Scientist question.

### 10.3 Critical Success Factors
1.  **Latency:** Speed is a feature.
2.  **Reliability:** The API must be up 99.9% of the time.

### 10.4 Further Reading
*   **Burkov:** "Machine Learning Engineering".
*   **Google:** "Hidden Technical Debt in Machine Learning Systems".

---

## Appendix

### A. Glossary
*   **Endpoint:** A specific URL where the API listens (e.g., `/predict`).
*   **Payload:** The data sent to the API.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **PSI** | $\sum (P_a - P_e) \ln(P_a/P_e)$ | Drift Detection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

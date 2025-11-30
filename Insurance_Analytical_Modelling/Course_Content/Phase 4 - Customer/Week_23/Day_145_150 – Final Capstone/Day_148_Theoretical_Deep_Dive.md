# Final Capstone: Deployment & MLOps (Part 4) - FastAPI & Monitoring - Theoretical Deep Dive

## Overview
"It runs on my machine" is not a deployment strategy.
In the Capstone, you must expose your model as a **Service**.
This day focuses on Containerization (Docker), Orchestration (Kubernetes), and Observability (Grafana).

---

## 1. Conceptual Foundation

### 1.1 The Inference Pattern

1.  **Real-Time (Online):** Single prediction in < 100ms. (e.g., Quote Generation).
    *   *Tech:* FastAPI, Flask.
2.  **Batch (Offline):** 1 million predictions overnight. (e.g., Churn Scoring).
    *   *Tech:* Airflow, Spark.
3.  **Streaming:** Event-driven. (e.g., Telematics Crash Detection).
    *   *Tech:* Kafka, Flink.

### 1.2 The "Shadow Mode" Strategy

*   **Concept:** Deploy the new model alongside the old one.
*   **Traffic:** 100% of traffic goes to *both*.
*   **Response:** Only the *Old Model* returns the response to the user.
*   **Analysis:** Compare the *New Model's* prediction to the *Actual Outcome* silently.
*   **Goal:** Prove safety before "Promoting" to live.

---

## 2. Mathematical Framework

### 2.1 Latency Budget

$$ \text{TotalLatency} = T_{\text{Network}} + T_{\text{FeatureStore}} + T_{\text{Model}} $$

*   *Constraint:* Total < 200ms.
*   *Network:* 50ms.
*   *Feature Store:* 10ms (Redis).
*   *Model:* 140ms remaining. (XGBoost is fine. BERT might be too slow).

### 2.2 Throughput (RPS)

$$ \text{RPS} = \frac{\text{Concurrency}}{\text{Latency}} $$

*   *Scenario:* 100 concurrent users. 0.2s latency.
*   *Capacity:* 500 Requests Per Second per Container.
*   *Scaling:* If you need 5,000 RPS, you need 10 Containers (Kubernetes HPA).

---

## 3. Theoretical Properties

### 3.1 Containerization (Docker)

*   **Problem:** "Dependency Hell". (Works with Pandas 1.0, breaks with Pandas 2.0).
*   **Solution:** **Docker**.
    *   *Dockerfile:* A recipe that builds an immutable Image.
    *   *Result:* If it runs in Docker on your laptop, it runs in Docker on the Cloud.

### 3.2 Orchestration (Kubernetes)

*   **Role:** The "Traffic Cop".
*   **Features:**
    *   **Self-Healing:** If a container crashes, restart it.
    *   **Auto-Scaling:** If CPU > 80%, add more pods.
    *   **Rolling Updates:** Update v1 to v2 without downtime.

---

## 4. Modeling Artifacts & Implementation

### 4.1 FastAPI Service (main.py)

```python
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import mlflow.xgboost
import pandas as pd

app = FastAPI()

# 1. Load Model (Global Scope)
model = mlflow.xgboost.load_model("models:/ChurnModel/Production")

class PredictionRequest(BaseModel):
    age: int
    premium: float
    claims_count: int

@app.post("/predict")
def predict(req: PredictionRequest):
    # 2. Preprocess
    df = pd.DataFrame([req.dict()])
    
    # 3. Predict
    try:
        prob = model.predict_proba(df)[0][1]
        return {"churn_probability": float(prob)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

### 4.2 Dockerfile

```dockerfile
# Base Image
FROM python:3.9-slim

# Working Directory
WORKDIR /app

# Dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Code
COPY main.py .

# Command
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]
```

---

## 5. Evaluation & Validation

### 5.1 Load Testing (Locust)

*   **Task:** Simulate 1,000 users hitting the API.
*   **Metric:** P99 Latency (The latency for the slowest 1% of users).
*   **Goal:** P99 < 500ms.

### 5.2 Observability (Prometheus + Grafana)

*   **Metrics to Track:**
    1.  **Request Count:** Is traffic spiking?
    2.  **Error Rate:** Are we returning 500s?
    3.  **Latency:** Is the model slow?
    4.  **Prediction Drift:** Is the average predicted risk shifting from 10% to 50%?

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Pickle" Security Risk**
    *   *Risk:* Loading a `.pkl` file from an untrusted source can execute arbitrary code (RCE).
    *   *Fix:* Only load models from your secure Model Registry (MLflow). Sign your artifacts.

2.  **Trap: Memory Leaks**
    *   *Scenario:* The API loads the model *inside* the request function.
    *   *Result:* The server runs out of RAM after 10 requests.
    *   *Fix:* Load the model *once* at startup (Global Variable).

---

## 7. Advanced Topics & Extensions

### 7.1 Canary Deployment

*   **Strategy:**
    *   Route 1% of traffic to v2.
    *   Route 99% of traffic to v1.
    *   Monitor v2 for errors.
    *   Gradually increase to 10%, 50%, 100%.

### 7.2 Serverless Inference (AWS Lambda)

*   **Pros:** Zero cost when idle. No servers to manage.
*   **Cons:** "Cold Start" latency (2-3 seconds) to load the model. Bad for real-time insurance quotes.

---

## 8. Regulatory & Governance Considerations

### 8.1 API Security

*   **Requirement:** Authentication.
*   **Standard:** API Keys or OAuth 2.0 (JWT Tokens).
*   **Rate Limiting:** Prevent DDoS attacks.

---

## 9. Practical Example

### 9.1 Worked Example: Deploying the "Fraud Scorer"

**Goal:** API that returns Fraud Score.
**Steps:**
1.  **Build:** `docker build -t fraud-scorer:v1 .`
2.  **Run:** `docker run -p 8000:80 fraud-scorer:v1`
3.  **Test:** `curl -X POST localhost:8000/predict -d '{"amount": 5000}'`
4.  **Deploy:** Push to AWS ECR. Update Kubernetes Deployment yaml.
5.  **Monitor:** Check Grafana dashboard for "High Fraud" alerts.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Containerize (Docker)** for consistency.
2.  **Expose (FastAPI)** for consumption.
3.  **Monitor (Grafana)** for sanity.

### 10.2 When to Use This Knowledge
*   **DevOps Interview:** "How do you handle model rollbacks?"
*   **Capstone Demo:** Live demo of the API using Swagger UI (`/docs`).

### 10.3 Critical Success Factors
1.  **Automation:** CI/CD pipeline (GitHub Actions) builds the Docker image automatically on `git push`.
2.  **Resilience:** The API should not crash if the Feature Store is down (Return a default value).

### 10.4 Further Reading
*   **Google:** "Hidden Technical Debt in Machine Learning Systems".

---

## Appendix

### A. Glossary
*   **HPA:** Horizontal Pod Autoscaler.
*   **JWT:** JSON Web Token.
*   **RPS:** Requests Per Second.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **P99 Latency** | 99th Percentile | Performance SLA |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*

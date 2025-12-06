# Day 153: The Ninja Mode: Shadow Deployments
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 22: Data & Model Quality

---

> **🎯 Focus Area:** You think your new model is great. But A/B testing is risky (real users might see errors). **Shadow Deployment** lets the new model process *real traffic* without the user ever knowing it exists.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** Traffic Mirroring (Shadowing) using Istio or Nginx.
2.  **Architect** an async comparison pipeline (Kafka -> Shadow Service -> Analysis).
3.  **Evaluate** Shadow results for "Prediction Parity" and "Latency Regression".
4.  **Detect** bugs that only appear in production scale (e.g., Memory Leaks).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine with `kubectl`.

### Software Environment
- Istio installed (or Nginx Ingress).

---

## 📖 Theoretical Foundation

### 1. The Mirroring Concept
*   **Request:** User sends `POST /predict`.
*   **Router:**
    1.  Sends to **Primary (V1)**. V1 response goes to User.
    2.  Sends copy to **Shadow (V2)**. V2 response is logged (fire & forget).
*   **Risk:** Zero impact on user response (unless mirroring adds latency/overhead to the router).
*   **Cost:** Double compute (Running V1 + V2).

### 2. What to Compare?
*   **Crashes:** Did V2 return 500?
*   **Latency:** Is V2 slower than V1?
*   **Value:** `diff(v1.pred, v2.pred)`. If difference is huge, investigate.

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Istio Mirroring

We assume a VirtualService routes to `v1`. We add `mirror` destination.

#### 📁 `manifests/shadow-rule.yaml`
```yaml
apiVersion: networking.istio.io/v1alpha3
kind: VirtualService
metadata:
  name: inference-service
spec:
  hosts:
  - my-model.com
  http:
  - route:
    - destination:
        host: model-v1
        subset: v1
      weight: 100
    mirror:
      host: model-v2
      subset: v2
    mirrorPercentage:
      value: 100 # Mirror 100% of traffic
```

### 👨‍💻 Infrastructure: Application-Level Shadowing (Python)

If you don't use Istio, do it in the Gateway.

#### 📁 `src/gateway_shadow.py`
```python
import asyncio
import httpx
from fastapi import FastAPI

app = FastAPI()

async def call_shadow(payload):
    try:
        async with httpx.AsyncClient() as client:
            # Fire and forget (don't await response body parse, but await send)
            # Actually, to truly not block, use create_task
            await client.post("http://model-v2/predict", json=payload)
    except Exception as e:
        # Log shadow failure but don't alarm
        print(f"Shadow failed: {e}")

@app.post("/predict")
async def predict(payload: dict):
    # 1. Trigger Shadow (Background Task)
    asyncio.create_task(call_shadow(payload))
    
    # 2. Call Primary
    async with httpx.AsyncClient() as client:
        resp = await client.post("http://model-v1/predict", json=payload)
        return resp.json()
```

### 👨‍💻 Core Implementation: The Shadow Analyzer

Process the logs from V1 and V2 to find mismatches.

#### 📁 `src/analyze_shadow.py`
```python
import pandas as pd
import numpy as np

# Load logs (Joined by Request ID)
# Request ID | V1_Pred | V2_Pred | V1_Latency | V2_Latency
df = pd.read_parquet("shadow_logs.parquet")

# 1. Error Check
v2_errors = df[df["v2_status"] == 500]
print(f"V2 Crashes: {len(v2_errors)}")

# 2. Latency Check
df["lat_diff"] = df["v2_latency"] - df["v1_latency"]
print(f"Avg Latency Increase: {df['lat_diff'].mean():.2f}ms")

# 3. Prediction Match
# For Regression: Correlation
corr = df["v1_pred"].corr(df["v2_pred"])
print(f"Prediction Correlation: {corr:.4f}")

# For Classification: Agreement Rate
agreement = (df["v1_class"] == df["v2_class"]).mean()
print(f"Agreement Rate: {agreement:.2%}")

# 4. Deep Dive Mismatches
mismatches = df[df["v1_class"] != df["v2_class"]]
print("Top Mismatches:")
print(mismatches.head())
```

---

## 🔬 Lab Exercise: "The Memory Leak"

### Task
Scale Test.
1.  Deploy V2 (with intentional bug: `GlobalList.append(data)`).
2.  Enable Shadowing (100%).
3.  Load Test the V1 endpoint.
4.  **Observation:**
    *   V1 Latency: Stable.
    *   V2 Latency: Increases slowly.
    *   Eventually V2 OOMs.
5.  **Impact:** User sees NOTHING. Zero outages.
6.  **Action:** Kill V2. Fix bug. Redeploy.

---

## 📖 Advanced Theory: Dark Launch
Use Shadowing to "prime" the cache.
*   New model V2 has empty cache.
*   Turn on Shadowing.
*   V2 computes predictions and populates Redis.
*   Once cache hit rate > 90%, switch traffic to V2 (Canary).
*   Prevents "Cold Start" latency spike.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Safety:** Shadowing is the safest deployment strategy. Canary is second. Blue/Green is third. Recreate is last.
2.  **Cost:** It requires 2x capacity. If you run 100 GPUs for V1, you need 100 GPUs for V2. Use Shadowing only for critical updates, or sample traffic (1%).
3.  **Side Effects:** Ensure V2 does NOT trigger side effects (e.g., Sending Emails, Charging Credit Cards). Pass a header `X-Shadow: true` and ensure downstream services mock the action.

### API Summary
```yaml
mirror:
  host: ...
```

---

**Day 153 Complete** ✅

*Next: Day 154 - Week 22 Review & Project - Quality Assurance System.*

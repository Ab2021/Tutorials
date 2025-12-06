# Day 204: Stress Fracture: Advanced Load Testing
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 30: Capstone Project Part 2

---

> **🎯 Focus Area:** "It works on my laptop" is irrelevant. "It works with 5 users" is boring. **Can it survive 100,000 users hitting the API at once?** Today we break the Titan Platform intentionally.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Deploy** a Distributed Locust Cluster (Master + 10 Workers) on Kubernetes.
2.  **Generate** the "Thundering Herd" traffic pattern to test Cold Start latency.
3.  **Identify** the Breaking Point (e.g., Ingress Controller CPU saturation).
4.  **Optimize** the `HPA` reaction time to handle spikes faster.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.
- Access to `titan-inf`.

### Software Environment
- `helm install locust`.

---

## 📖 Theoretical Foundation

### 1. Types of Tests
*   **Load Test:** Sustained traffic (e.g., 100 RPS for 1 hour).
*   **Stress Test:** Increasing traffic until failure (e.g., 100 -> 1000 -> Crash).
*   **Spike Test:** Sudden burst (e.g., 0 -> 500 in 1 second). "Thundering Herd".
*   **Soak Test:** Moderate traffic for 24 hours (Detect Memory Leaks).

### 2. Distributed Load
A single laptop can simulate ~500 users.
To simulate 50k users, you need a Cluster of Load Generators.
**Locust Architecture:** master (Aggregates stats) + workers (Generate requests).

---

## 💻 Implementation

### 👨‍💻 Infrastructure: Distributed Locust

Use Helm to deploy the attackers.

#### 📁 `manifests/loadtest/locust-values.yaml`
```yaml
loadtest:
  name: titan-stress
  locust_locustfile_configmap: "titan-locustfile"
  
service:
  type: LoadBalancer

worker:
  replicas: 10 # 10 Attackers
  resources:
    limits:
      cpu: 1
      memory: 1Gi
```

### 👨‍💻 Core Implementation: The Locustfile

Define User Behavior.

#### 📁 `manifests/loadtest/locustfile.py`
```python
from locust import HttpUser, task, between
import random
import json

class TitanUser(HttpUser):
    wait_time = between(1, 5) # User thinks for 1-5s

    @task(3)
    def predict_bert(self):
        # 70% of traffic: BERT (Heavy)
        payload = {"inputs": [[random.randint(0, 100) for _ in range(128)]]}
        self.client.post(
            "/v1/models/bert:predict",
            json=payload,
            headers={"Authorization": "Bearer ..."}
        )

    @task(1)
    def health_check(self):
        # 30% of traffic: Health (Light)
        self.client.get("/v1/models/bert")
```

### 👨‍💻 Core Implementation: The Spike Script (K6)

Simulate a Marketing Push.

#### 📁 `manifests/loadtest/spike.js`
```javascript
import http from 'k6/http';
import { sleep } from 'k6';

export const options = {
  stages: [
    { duration: '10s', target: 10 },   // Warmup
    { duration: '10s', target: 2000 }, // SPIKE! 10 -> 2000 users
    { duration: '1m', target: 2000 },  // Sustain
    { duration: '10s', target: 0 },    // Cooldown
  ],
};

export default function () {
  const payload = JSON.stringify({ inputs: [[1, 2, 3]] });
  const params = { headers: { 'Content-Type': 'application/json' } };
  http.post('https://bert.titan.ai/v1/models/bert:predict', payload, params);
  sleep(1);
}
```

---

## 🔬 Lab Exercise: "The 502 Bad Gateway"

### Task
Find the Bottle Neck.
1.  **Run:** Locust with 100 users.
    *   Result: `200 OK`. Latency 100ms.
2.  **Run:** Locust with 1000 users.
    *   Result: `200 OK`. Latency 500ms. (Autoscaler adds Pods).
3.  **Run:** Locust with 5000 users.
    *   Result: `502 Bad Gateway`.
    *   Observation: KServe Pods are healthy.
    *   **Root Cause:** The Ingress Controller (Nginx/Istio Gateway) CPU hit 100%. It couldn't route the requests.
    *   **Fix:** Scale Istio Gateway from 2 to 10 replicas.

---

## 📖 Advanced Theory: Rate Limiting
To prevent the "Death Spiral" (where retries kill the server), you must reject excess traffic efficiently.
**Envoy Rate Limiting:**
*   Global Limit: 1000 RPS.
*   Per Source IP: 10 RPS.
*   Per JWT User: 50 RPS.
*   **Response:** `429 Too Many Requests`. This is better than `502` because the client knows to back off.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Test inside the VPC:** Running load tests from your laptop over the internet bottlenecks on *your* Wifi. Run Locust *inside* the cluster (ClusterIP) to test the Service Capacity, not the Internet Bandwidth.
2.  **Cold Starts Kill:** In a Spike Test, the first 1 minute usually has 90% failure rate because Autoscaler is reacting. Fix: Predict traffic or Over-provision (MinReplicas=50).
3.  **Database Locking:** Often the ML Model is fine, but the *Logging* of the prediction (to Postgres) locks the DB table. Use Async Logging (Kafka).

### API Summary
```bash
k6 run spike.js
```

---

**Day 204 Complete** ✅

*Next: Day 205 - Cost Optimization V2 - The Spot Market.*

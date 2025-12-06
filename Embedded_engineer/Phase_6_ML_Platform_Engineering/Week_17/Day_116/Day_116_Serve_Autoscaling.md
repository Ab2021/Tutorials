# Day 116: Elastic Inference: Autoscaling Ray Serve
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** Static provisioning for inference is expensive. Traffic is bursty. **Ray Serve Autoscaling** dynamically adjusts replica counts based on request queue depth.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** `autoscaling_config` with min/max limits.
2.  **Tune** `target_num_ongoing_requests_per_replica` for responsiveness.
3.  **Implement** "Scale to Zero" for rarely used models.
4.  **Stress Test** a deployment to trigger a scale-up event.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install locust`. (Load Generator).

---

## 📖 Theoretical Foundation

### 1. The Metric
Unlike K8s HPA (CPU/Memory), Ray Serve scales on **Queued Requests**.
*   **Target:** `10`.
*   **Scenario:** 1 Replica. 50 requests arrive. 
*   **Logic:** $50 / 10 = 5$ replicas needed.
*   **Action:** Controller starts 4 new replicas immediately.

### 2. Smoothness
*   **Upscale Delay:** How fast to react? (Default: fast).
*   **Downscale Delay:** How long to wait before killing? (Prevent flapping).
*   **Lookback Period:** Smoothing window.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Autoscaled Deployment

#### 📁 `src/04_autoscale.py`
```python
import ray
from ray import serve
import time

@serve.deployment(
    autoscaling_config={
        "min_replicas": 1,
        "max_replicas": 5,
        "target_num_ongoing_requests_per_replica": 2,
    }
)
class Sleeper:
    def __call__(self):
        # Simulate work to keep the request "active"
        time.sleep(1)
        return "Done"

sleeper = Sleeper.bind()

if __name__ == "__main__":
    serve.run(sleeper)
    
    # Generate Load
    handle = serve.get_app_handle("default")
    
    # 1. Fire 20 async requests
    print("Firing requests...")
    refs = [handle.remote() for _ in range(20)]
    
    # 2. Watch Dashboard
    # Request depth = 20. Target = 2.
    # Desired Replicas = 10.
    # Max capped at 5.
    # Expect: Replicas scale to 5.
    
    print(ray.get(refs))
```

### 👨‍💻 Lab: Scale to Zero

Change `min_replicas=0`.
1.  Wait 1 minute. Replicas -> 0.
2.  Send 1 request.
3.  **Behavior:** Request blocks.
4.  **Controller:** Sees queue > 0. Starts 1 Replica (Cold Start).
5.  **Replica:** Init.
6.  **Request:** Processed.

---

## 🔬 Lab Exercise: "Locust Attack"

### Task
Use Locust.
1.  Create `locustfile.py` hitting your endpoint.
2.  Ramp users from 1 to 100.
3.  Monitor Ray Dashboard -> "Serve" tab.
4.  **Observation:** You should see a staircase graph of Replica Count following the User Count.

---

## 📝 Daily Summary

### Key Takeaways
1.  **GPU Cold Starts:** Scaling a GPU model to zero saves huge money, but initialization (loading weights to VRAM) takes seconds. Use only for internal/async tooling, not user-facing latencies.
2.  **Concurrency:** Ensure your `target` matches your concurrency capability. If your model processes 1 req/sec, setting target=100 will cause massive latency before scaling triggered.
3.  **Knative:** Similar to Knative, but native to the Ray ecosystem so it works with the Pipeline graph.

### API Summary
```python
autoscaling_config={"min_replicas": 0}
```

---

**Day 116 Complete** ✅

*Next: Day 117 - Request Batching - Improving Throughput.*

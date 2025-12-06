# Day 118: Safety First: Canary Rollouts
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** You pushed a new model. It has higher accuracy but 5x latency. Detecting this before it hits 100% of users is vital. Use **Traffic Splitting** and **Canary Rollouts**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Implement** a Router/Splitter pattern in the Ingress deployment.
2.  **Perform** a Gradual Rollout (10% -> 50% -> 100%) by updating the config.
3.  **Use** `user_config` to change application behavior dynamically (e.g., feature flags).
4.  **Execute** a Shadow Deployment (Run V2 but discard result) for verification.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. In-Code vs Infrastructure Splitting
*   **Infrastructure (Istio/RayService):** Shifts traffic at the K8s Service level. Good for shifting entire Graphs.
*   **In-Code (Ingress Logic):** Shifts traffic *inside* the graph. Good for shifting between two specific nodes (ModelA vs ModelB).

### 2. User Config
Every deployment has a `reconfigure` method.
*   Update YAML: `user_config: { "split_ratio": 0.1 }`
*   Ray Serve: Calls `deployment.reconfigure({'split_ratio': 0.1})`.
*   Ingress: Starts routing 10% to V2.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Router

#### 📁 `src/06_canary.py`
```python
import ray
from ray import serve
import random

@serve.deployment
class ModelV1:
    def __call__(self):
        return "V1 (Stable)"

@serve.deployment
class ModelV2:
    def __call__(self):
        return "V2 (Experimental)"

@serve.deployment(user_config={"v2_percent": 0.0})
class Router:
    def __init__(self, v1, v2):
        self.v1 = v1
        self.v2 = v2
        self.percent = 0.0

    def reconfigure(self, config):
        # Called when we update the YAML
        self.percent = config["v2_percent"]
        print(f"Updated split to: {self.percent}")

    async def __call__(self):
        if random.random() < self.percent:
            return await self.v2.remote()
        return await self.v1.remote()

# Wire
v1 = ModelV1.bind()
v2 = ModelV2.bind()
app = Router.bind(v1, v2)

# Run
if __name__ == "__main__":
    serve.run(app)
    
    # Simulate client
    handle = serve.get_app_handle("default")
    print(ray.get(handle.remote())) # V1
```

### 👨‍💻 Lab: Rolling Output

1.  Run the code. Output is always "V1".
2.  Update Config via Python SDK (Or REST API):
    ```python
    serve.run(app, name="default", route_prefix="/")
    # Actually, in code we can't easily update user_config for running bind?
    # We simulate it by redeploying the specific deployment logic or updating via HTTP.
    ```
    *Better approach:* Use the `serve build` approach to generate YAML, edit YAML, `serve deploy`.

#### 📁 `config.yaml`
```yaml
applications:
- name: default
  import_path: src.06_canary:app
  deployments:
  - name: Router
    user_config:
      v2_percent: 0.5
```
Command: `serve deploy config.yaml`.
**Result:** 50% traffic goes to V2.

---

## 🔬 Lab Exercise: "Shadow Mode"

### Task
Implement Shadowing.
1.  Modify Router:
    ```python
    async def __call__(self):
        ret = await self.v1.remote() # Always return V1
        
        # Fire V2 but ignore result (Shadow)
        # Don't await it to avoid latency impact? 
        # Actually in Python, if we don't await, we must ensure it runs.
        # Use asyncio.create_task()
        self.v2.remote() 
        
        return ret
    ```
2.  **Benefit:** Test V2 on real production traffic without user impact.

---

## 📝 Daily Summary

### Key Takeaways
1.  **State Management:** `reconfigure` updates state without restarting the actor replica (hot reload).
2.  **Granularity:** You can have multiple experiments running (V2, V3, V4) routed by user ID.
3.  **Safety:** If `reconfigure` raises an exception, Ray Serve rejects the config update and keeps the old state.

### API Summary
```python
def reconfigure(self, config): ...
```

---

**Day 118 Complete** ✅

*Next: Day 119 - Week 17 Review & Project - Building a Scalable RAG Service.*

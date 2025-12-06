# Day 113: The Microservices Engine: Ray Serve
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** Flask is great for demos, but it doesn't pipeline requests or batch inputs natively. **Ray Serve** provides a scalable, framework-agnostic serving layer built on top of Ray Actors.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Compare** Ray Serve with standard tools (FastAPI, TorchServe, Triton).
2.  **Create** a basic `@serve.deployment` class.
3.  **Deploy** the service via `serve.run()` (Local Developer Mode).
4.  **Invoke** the service via HTTP and Python `ServeHandle`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `pip install "ray[serve]" requests`.

---

## 📖 Theoretical Foundation

### 1. The Architecture
*   **Controller:** Global manager. Handles autoscaling, config updates.
*   **Proxy:** The HTTP Ingress. Runs on the Head Node (and others). Routes `/foo` to the correct actor.
*   **Replica:** An Actor running your code.
*   **Benefit:** Unlike microservices where Service A talks to Service B via JSON/HTTP, Ray Serve allow Service A to talk to Service B via **Shared Memory Objects** (Zero serialization overhead).

### 2. Integration
Ray Serve is framework agnostic. You can serve PyTorch, Scikit-Learn, or just plain Python logic.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Hello World

#### 📁 `src/01_serve_basic.py`
```python
import ray
from ray import serve
import requests

# 1. Define Deployment
@serve.deployment(num_replicas=2)
class Translator:
    def __init__(self):
        # Load heavy models here (runs once per replica)
        print("Loading Model...")
        self.model = lambda x: x + " -> Bonjour"

    # The __call__ method handles the request
    def __call__(self, request):
        if isinstance(request, str):
            text = request
        else:
            # Assume HTTP Request object
            text = request.query_params["text"]
        
        return self.model(text)

# 2. Bind and Run
# In production, we assume 'ray start' is running.
ray.init()
serve.start(detached=True) # detached=True means it survives driver exit

# Deploy
Translator.deploy()

# 3. Test - HTTP
print("Testing HTTP...")
resp = requests.get("http://127.0.0.1:8000/Translator?text=Hello")
print(resp.text)

# 4. Test - Python Handle (Faster)
print("Testing Handle...")
handle = Translator.get_handle()
ref = handle.remote("Hello")
print(ray.get(ref))
```

### 👨‍💻 Core Implementation: Argument Packing

Ray Serve wraps `FastAPI`. You can type-hint inputs.

```python
from fastapi import Request

@serve.deployment
class Advanced:
    async def __call__(self, http_request: Request):
        json_body = await http_request.json()
        return {"count": len(json_body["text"])}
```

---

## 🔬 Lab Exercise: "Resource Isolation"

### Task
Assign GPUs.
1.  Change decorator: `@serve.deployment(ray_actor_options={"num_gpus": 0.5})`.
2.  Deploy.
3.  Check Ray Dashboard. You will see 2 Actor Replicas claiming GPU memory.
4.  **Note:** Requests are load-balanced Round-Robin by default (or Power of Two Choices).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Init:** `__init__` is for heavy lifting (loading weights). `__call__` is for inference.
2.  **Handles:** Use `ServeHandle` for service-to-service communication. It is significantly faster than `requests.get("localhost")`.
3.  **Port:** Default is 8000. Configurable via `serve.start(http_options={"port": 9000})`.

### API Summary
```python
serve.run(entrypoint)
serve.shutdown()
```

---

**Day 113 Complete** ✅

*Next: Day 114 - Deployments & Ingress - Managing complicated APIs.*

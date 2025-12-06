# Day 115: The Assembly Line: Model Composition
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** Real-world AI is rarely a single model. It is a pipeline: Preprocessing -> Embedding -> Retrieval -> Re-ranking -> Generation. Ray Serve composes these steps elegantly.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Design** a Graph of deployments (DAG).
2.  **Implement** an Ensembling pattern (Fan-out/Fan-in).
3.  **Trace** the execution flow of a single request across multiple actors.
4.  **Optimize** the pipeline by independently scaling bottlenecks.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Deployment Graph
In standard microservices, mapping the dependency graph is hard (Need distributed tracing).
In Ray Serve, the graph is explicit in the code structure (Handles).
*   **Binding:** `a = A.bind()`, `b = B.bind(a)`. This defines the topology.
*   **Execution:** Ray handles the placement. A and B can be on different nodes.

### 2. Common Patterns
*   **Preprocessing:** `Input -> CV2-Resize -> ResNet`.
*   **Ensemble:** `Input -> [ModelV1, ModelV2] -> Average`.
*   **RAG:** `Input -> Embedder -> VectorDB -> LLM`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: The Ensemble

#### 📁 `src/03_pipeline.py`
```python
import ray
from ray import serve
import asyncio

@serve.deployment
class ModelA:
    def __call__(self, x: int):
        return x * 2

@serve.deployment
class ModelB:
    def __call__(self, x: int):
        return x + 5

@serve.deployment
class Ensemble:
    def __init__(self, m1, m2):
        self.m1 = m1
        self.m2 = m2

    async def __call__(self, x: int):
        # Fan-out: Call both in parallel
        # Note: We must use asyncio.gather for true parallelism
        curr_fut1 = self.m1.remote(x)
        curr_fut2 = self.m2.remote(x)
        
        # Await both
        res1, res2 = await asyncio.gather(curr_fut1, curr_fut2)
        
        # Fan-in: Average
        return (res1 + res2) / 2

# Wiring
m1 = ModelA.bind()
m2 = ModelB.bind()
ensemble = Ensemble.bind(m1, m2)

# Run
if __name__ == "__main__":
    serve.run(ensemble)
    # Test via handle in another script or simple curl if Ingress added
    # For now, let's just use Python API
    handle = serve.get_app_handle("default")
    print(ray.get(handle.remote(10)))
    # ModelA: 20. ModelB: 15. Avg: 17.5.
```

### 👨‍💻 Lab: Scaling Bottlenecks

1.  Assume `ModelA` is slow (resizes images).
2.  Assume `ModelB` is fast.
3.  We can scale just A:
    ```python
    @serve.deployment(num_replicas=5)
    class ModelA: ...
    ```
4.  The `Ensemble` stays at 1 replica. It load-balances requests to the 5 `ModelA` replicas.

---

## 🔬 Lab Exercise: "Dead Letter Queue"

### Task
Handle Failures.
1.  Make `ModelA` raise Error occasionally.
2.  In `Ensemble`, wrap calls in `try/catch`.
3.  If A fails, maybe just return B's result (Degraded mode).
4.  **Insight:** This logic lives in the Supervisor/Ensemble actor, giving you fine-grained control over failure modes.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Shared Memory:** Data passing between A and B uses Zero-Copy Plasma if on the same node. This is much faster than HTTP over localhost.
2.  **Versioning:** You can bind `ModelV1` and `ModelV2` in the same graph for A/B testing logic inside the Ingress.
3.  **Drivers:** The `bind()` syntax is the modern (2.0+) way to define applications. It decouples definition from execution.

### API Summary
```python
asyncio.gather(ref1, ref2)
```

---

**Day 115 Complete** ✅

*Next: Day 116 - Autoscaling Serve - Traffic-based scaling.*

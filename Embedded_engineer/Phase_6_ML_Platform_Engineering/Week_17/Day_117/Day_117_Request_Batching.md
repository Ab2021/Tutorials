# Day 117: Throughput Monster: Request Batching
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 17: Ray Serve - Model Serving

---

> **🎯 Focus Area:** Sending 1 sample at a time to a GPU is like using a Ferrari to deliver a single pizza. **Dynamic Batching** automatically groups independent requests into batches to maximize GPU saturation.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Transform** a scalar endpoint into a batched endpoint using `@serve.batch`.
2.  **Tune** `max_batch_size` and `batch_wait_timeout_s` for the Latency vs Throughput trade-off.
3.  **Process** a list of inputs using vectorized operations (NumPy/PyTorch).
4.  **Demonstrate** the throughput increase (Requests Per Second) under load.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine (GPU beneficial).

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Trade-off
*   **No Batching:** Latency = Inference Time. Throughput = 1 / InfTime.
*   **Batching:** Latency = Wait Time + Batch Inf Time. Throughput = Batch Size / (Wait + Batch Inf).
*   **Goal:** Increase Batch Size until "Batch Inf Time" starts growing linearly.

### 2. Implementation Logic
1.  Decorator `@serve.batch`.
2.  Change `__call__(self, request)` to `__call__(self, requests: List[Request])`.
3.  **Crucial:** You MUST return a List of equal length. Ray maps `Result[i]` to `Request[i]`.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Batched Adder

#### 📁 `src/05_batching.py`
```python
import ray
from ray import serve
import time
import asyncio

@serve.deployment
class BatchedModel:
    def __init__(self):
        print("Model Init")

    @serve.batch(max_batch_size=8, batch_wait_timeout_s=0.1)
    async def handle_batch(self, inputs: list):
        # inputs is a list of whatever was passed to handle.remote()
        print(f"Processing batch of size {len(inputs)}")
        
        # Vectorized Operation (Simulated)
        # In PyTorch: torch.stack(inputs).forward()
        results = [x * 2 for x in inputs]
        
        return results

    async def __call__(self, request: int):
        # Hook the batch handler
        return await self.handle_batch(request)

model = BatchedModel.bind()

if __name__ == "__main__":
    serve.run(model)
    handle = serve.get_app_handle("default")
    
    # Fire 100 requests
    futures = [handle.remote(i) for i in range(20)]
    
    # Expect: ~3 batches of size 8
    print(ray.get(futures))
```

### 👨‍💻 Core Implementation: PyTorch Integration

```python
    @serve.batch(max_batch_size=4, batch_wait_timeout_s=0.1)
    async def handle_batch(self, inputs: list):
        # 1. Preprocess
        tensor_batch = torch.tensor(inputs) # Stack
        
        # 2. Inference
        with torch.no_grad():
            output_batch = self.model(tensor_batch) # Fast!
            
        # 3. Postprocess
        return output_batch.tolist()
```

---

## 🔬 Lab Exercise: "Tuning Window"

### Task
Measure Latency.
1.  `batch_wait_timeout_s=1.0` (Too high).
2.  Send 1 request.
3.  **Observation:** Client waits 1 second plus inference time. Bad UX.
4.  Set `batch_wait_timeout_s=0.01` (10ms).
5.  Send 100 requests.
6.  **Observation:** If load is high, batches fill up INSTANTLY (don't wait 10ms). The timeout is only for low load.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Simplicity:** You don't need a separate "Batching Proxy". Code it directly in the deployment.
2.  **Async:** Batching requires `async def` because the actor needs to yield control to the loop to accept the next request into the buffer.
3.  **GPU Utilization:** Look at `nvidia-smi`. With batching, Volatile-GPU-Util should stay high/stable, not spikey.

### API Summary
```python
@serve.batch
```

---

**Day 117 Complete** ✅

*Next: Day 118 - Production Rollouts - Canary releases in Ray Serve.*

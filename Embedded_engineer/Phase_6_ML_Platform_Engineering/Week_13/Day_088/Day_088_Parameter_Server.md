# Day 88: The Parameter Server Pattern
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** Before automated frameworks (Ray Train), we built systems manually. Implementing a **Parameter Server (PS)** from scratch teaches you exactly how distributed state synchronization works.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Diagram** the flow of gradients in a Parameter Server architecture.
2.  **Implement** a Sharded Parameter Server using Ray Actors.
3.  **Contrast** Synchronous SGD vs Asynchronous SGD.
4.  **Handle** "Stale Gradients" in async training.
5.  **Use** `ray.put` to broadcast weights efficiently.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`, `numpy`.

---

## 📖 Theoretical Foundation

### 1. The Bottleneck
If you have 10GB of weights, and 10 workers:
*   Every step, 10 workers push 10GB gradients -> 1 Server.
*   Server applies update.
*   Server pushes 10GB new weights -> 10 Workers.
**Network Traffic:** 200GB per step.
**Solution:** Sharding. Split weights across multiple PS Actors (PS-0 holds Layer 1-10, PS-1 holds Layer 11-20).

### 2. Async Training (Hogwild)
Workers don't wait for each other.
*   Worker 1 computes gradient on weights $W_{t}$.
*   Worker 2 commits update $W_{t+1}$.
*   Worker 1 commits update based on $W_{t}$ (Stale!).
*   **Result:** Faster (no stragglers), but effectively adding noise to training.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Simple Sync Parameter Server

#### 📁 `src/05_param_server.py`
```python
import ray
import numpy as np
import time

ray.init()

@ray.remote
class ParameterServer:
    def __init__(self, dim):
        self.weights = np.zeros(dim)
    
    def get_weights(self):
        return self.weights
    
    def apply_gradients(self, gradients):
        # Apply sum of gradients
        # In real life: Optimizer.step
        total_grad = sum(gradients)
        self.weights -= 0.01 * total_grad
        return self.weights

@ray.remote
class Worker:
    def __init__(self, id):
        self.id = id
    
    def compute_gradient(self, weights):
        # Simulate work: grad = (weights - target)
        # Target is 1.0
        time.sleep(0.1)
        return (weights - 1.0) 

# Init
dim = 10
ps = ParameterServer.remote(dim)
workers = [Worker.remote(i) for i in range(4)]

# Training Loop (Synchronous)
for epoch in range(5):
    # 1. Broadcast weights
    # Better: current_weights_ref = ps.get_weights.remote()
    current_weights = ray.get(ps.get_weights.remote())
    
    # 2. Put weights in object store for zero-copy access by workers
    weights_ref = ray.put(current_weights)
    
    # 3. Compute Gradients
    grad_futures = [w.compute_gradient.remote(weights_ref) for w in workers]
    gradients = ray.get(grad_futures)
    
    # 4. Update
    ps.apply_gradients.remote(gradients)
    
    print(f"Epoch {epoch} complete. Weights[0]: {current_weights[0]:.3f}")
```

### 👨‍💻 Optimization: Tree Reduce
Instead of sending all gradients to PS, workers can aggregate locally or hierarchically (AllReduce concept) before network transfer. Ray supports `tree_reduce`.

---

## 🔬 Lab Exercise: "Async Chaos"

### Task
Modify the loop to be Asynchronous.
1.  Worker fetches weights.
2.  Worker calculates grad.
3.  Worker pushes grad.
4.  **Crucially:** Worker does NOT wait for other workers.
5.  **Code Hint:** Use a `while` loop with `ray.wait(futures)` to process whoever finishes first.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Centralization vs Decentralization:** PS is Centralized (Hub & Spoke). `AllReduce` (NCCL) is Decentralized (Ring). `AllReduce` is generally faster for dense GPU training. PS is better for sparse models (Embedding layers in Recommender Systems).
2.  **Network Bound:** This pattern is heavily network bound. Use 100Gbps interfaces.
3.  **Ray's Role:** Ray makes writing this infrastructure trivial (50 lines of Python) compared to writing C++ MPI code or manual gRPC.

### API Summary
```python
ray.wait(futures, num_returns=1) # Return as soon as 1 task finishes
```

---

**Day 88 Complete** ✅

*Next: Day 89 - Fault Tolerance - Handling dead actors and node failures.*

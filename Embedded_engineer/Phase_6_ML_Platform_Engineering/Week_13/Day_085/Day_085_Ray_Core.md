# Day 85: The Unified Compute Framework: Ray
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** Python's `multiprocessing` library is stuck on a single machine. **Ray** breaks that barrier, allowing you to parallelize functions (Tasks) and classes (Actors) across thousands of nodes with zero code changes.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Explain** the limitations of Python `multiprocessing` regarding pickling and cluster scaling.
2.  **Initialize** a local Ray cluster using `ray.init()`.
3.  **Convert** a synchronous Python function into a distributed **Ray Task** (`@ray.remote`).
4.  **Create** a distributed **Ray Actor** to manage state (e.g., a counter or model holder).
5.  **Visualize** the execution tasks using the Ray Dashboard.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local machine (Ray starts a mini-cluster in background).

### Software Environment
- `pip install ray[default]`.

---

## 📖 Theoretical Foundation

### 1. The Global Interpreter Lock (GIL) & Scale
Python is single-threaded. `multiprocessing` works by spawning processes, but:
*   Processes cannot easily share memory (expensive IPC).
*   Processes cannot span multiple servers.
*   Error handling is terrible.

### 2. The Ray Engine
Ray introduces a **Distributed Scheduler** and **Object Store (Plasma)**.
*   **Task:** A function invocation. Stateless. Returns a `Future` (ObjectRef).
*   **Actor:** A Class instance. Stateful. Lives on a specific node. Methods are messages sent to it.
*   **Object Store:** Implementation of shared memory. Task A writes output; Task B reads it. Zero-copy on local node; automatic network transfer across nodes.

### 3. Syntax Magic
*   `func()` -> Synchronous.
*   `@ray.remote` -> Decorator.
*   `func.remote()` -> Asynchronous call. Returns `ObjectRef`.
*   `ray.get(ref)` -> Blocking wait for result.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Parallel Tasks

#### 📁 `src/01_ray_tasks.py`
```python
import ray
import time
import math

# 1. Start Ray (Local Cluster)
# auto-detects CPUs.
ray.init()

# 2. Define a Task
@ray.remote
def slow_square(x):
    time.sleep(1) # Simulate heavy work
    return x * x

# 3. Synchronous (Slow)
start = time.time()
results_sync = [slow_square.func(i) for i in range(4)]
print(f"Sync took: {time.time() - start:.2f}s") # ~4.0s

# 4. Asynchronous (Fast)
start = time.time()
# Launch 4 tasks in parallel
futures = [slow_square.remote(i) for i in range(4)]
# Block until all done
results_async = ray.get(futures)
print(f"Async took: {time.time() - start:.2f}s") # ~1.0s (if you have 4 cores)

print(results_async)
```

### 👨‍💻 Core Implementation: Stateful Actors

Use Actors when you need to hold resources (like a GPU Model or a Database Connection).

#### 📁 `src/02_ray_actors.py`
```python
import ray

ray.init()

@ray.remote
class Counter:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
        return self.value

    def get_value(self):
        return self.value

# Create an Actor instance (Process)
counter = Counter.remote()

# Call methods (Tasks sent to the actor)
# These execute serially on the actor process ( Thread-safe! )
f1 = counter.increment.remote()
f2 = counter.increment.remote()

print(ray.get([f1, f2])) # [1, 2]
```

### 👨‍💻 Accessing the Dashboard

When `ray.init()` runs, it prints:
`View the Ray dashboard at http://127.0.0.1:8265`
Open it. usage graphs, log viewers, and memory maps.

---

## 🔬 Lab Exercise: "Pi Estimation"

### Task
Monte Carlo Pi Estimation using Ray.
1.  Define a task `sample(n)` that generates `n` random points and counts how many are inside the unit circle.
2.  Launch 10 tasks with `n=1,000,000`.
3.  Sum the results using `ray.get`.
4.  Calculate Pi.
5.  **Challenge:** Compare 1 worker vs 8 workers.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Lazy Evaluation:** `func.remote()` returns immediately. Computation happens in background. `ray.get()` blocks.
2.  **State Management:** Use Tasks for data processing (Map/Reduce). Use Actors for Services (Model Server, Parameter Server, Environment Simulator).
3.  **Cluster Transparency:** The code `src/01_ray_tasks.py` runs on your laptop. If you connect it to a KubeRay cluster (Week 14), it runs on 100 nodes *without changing a line of code*.

### API Summary
```python
ray.init()
ray.remote
ray.put() # Store object in shared memory
ray.get() # Retrieve object
```

---

**Day 85 Complete** ✅

*Next: Day 86 - Ray Object Store & Memory Management - Zero-Copy Data Handling.*

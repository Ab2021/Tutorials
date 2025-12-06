# Day 89: Chaos Engineering: Fault Tolerance in Ray
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** In a cluster of 1000 GPUs, hardware failure is a daily event. Ray provides built-in mechanisms to handle **Task Retries**, **Actor Restarts**, and **Lineage Reconstruction**.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** `max_retries` for Tasks to handle transient network errors.
2.  **Use** `max_restarts` for Actors to survive node crashes.
3.  **Demonstrate** Ray's Lineage Reconstruction capabilities (re-computing lost objects).
4.  **Implement** Checkpointing for Actors so they wake up with memory restored.
5.  **Simulate** a node failure using `sys.exit()`.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Lineage Graph
Ray remembers: "Object C was created by `funcB(Object A)`. Object A was created by `funcA()`."
*   If the node holding Object C dies...
*   Ray checks: Do I have Object A?
    *   Yes: Re-run `funcB`.
    *   No: Re-run `funcA`, then `funcB`.
*   **Result:** Self-healing, *as long as tasks are deterministic*.

### 2. Actor Resilience
Actors are Stateful. Lineage cannot easily restore `self.count += 1` if we don't know how many times it was called.
*   **Restarting:** Ray can create a fresh actor on a new node.
*   **Checkpointing:** You must save state to S3/Disk periodically. Upon restart, the `__init__` should load the checkpoint.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Unstable Tasks

#### 📁 `src/06_faulty_task.py`
```python
import ray
import sys
import random

ray.init()

@ray.remote(max_retries=3)
def unstable_math(x):
    if random.random() < 0.5:
        print("Simulating crash!")
        sys.exit(1) # Kill the worker process
    return x * x

# Launch it. It might crash 1 or 2 times, but Ray retries.
try:
    print(ray.get(unstable_math.remote(10)))
except ray.exceptions.WorkerCrashedError:
    print("Failed even after retries.")
```

### 👨‍💻 Core Implementation: Resilient Actor

```python
@ray.remote(max_restarts=5)
class Counter:
    def __init__(self):
        self.value = 0
        self.load_checkpoint()

    def increment(self):
        self.value += 1
        # In real life: Save to disk every N steps
        if self.value % 10 == 0:
            self.save_checkpoint()
        
        # Simulate Crash
        if self.value == 15:
            sys.exit(1)
            
        return self.value

    def save_checkpoint(self):
        with open("checkpoint.dat", "w") as f:
            f.write(str(self.value))

    def load_checkpoint(self):
        try:
            with open("checkpoint.dat", "r") as f:
                self.value = int(f.read())
                print(f"Restored state: {self.value}")
        except FileNotFoundError:
            pass

counter = Counter.remote()
for i in range(20):
    try:
        print(ray.get(counter.increment.remote()))
    except Exception as e:
        print(f"Actor crashed! Ray is restarting it... {e}")
        # Note: We must retry the failed call in application logic usually
        # or use specialized libraries that handle invocation retries
```

---

## 🔬 Lab Exercise: "Kill the Node"

### Task
Simulate Object Loss.
1.  Run a task that creates a 1GB object on Node A.
2.  Store the ref in Driver.
3.  Kill Node A (or just the worker process).
4.  Try `ray.get(ref)`.
5.  **Observation:** Ray notices the object is gone. It sees the task that created it. It re-runs the task on Node B. The `ray.get` call blocks until the re-computation finishes, then returns success.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Immutability wins:** Lineage reconstruction only works seamlessly for Tasks because they are pure functions (conceptually).
2.  **Actor State:** Ray cannot magically restore `self.value`. Applications must implement checkpoint logic.
3.  **Head Node:** If the Head Node (GCS) dies, the whole Cluster dies (usually). High Availability (HA) for Ray Head is enabled by KubeRay (Week 14).

### API Summary
```python
@ray.remote(retry_exceptions=True) # Catch app-level exceptions too
```

---

**Day 89 Complete** ✅

*Next: Day 90 - Ray Runtime Environments - Shipping dependencies dynamically.*

# Day 87: The Tetris Player: Ray Scheduling
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** A distributed system is useless if it puts your Training Worker on a CPU node. Learn how to use **Ray Resources** and **Placement Groups** to control exactly where your code runs.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Define** and request logical resources (`num_cpus`, `num_gpus`).
2.  **Explain** Fractional GPU Scheduling and how Ray manages `CUDA_VISIBLE_DEVICES`.
3.  **Create** Custom Resources (e.g., `TPU: 4`) for specialized hardware.
4.  **Implement** Placement Groups for Gang Scheduling (All-or-Nothing).
5.  **Debug** "Pending" tasks due to resource starvation.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local machine (CPU only is fine, we can simulate GPUs).

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. Logical vs Physical Resources
Ray is a **Logical Scheduler**.
*   **Init:** `ray.init(num_cpus=4, num_gpus=2)`. This tells Ray "I have these resources".
*   **Request:** `@ray.remote(num_gpus=1)`.
*   **Effect:** Ray decrements the counter.
*   **Enforcement:** Ray sets `CUDA_VISIBLE_DEVICES` env var for the worker process. It does *not* enforce strict limits via cgroups (K8s does that).

### 2. Fractional GPUs
Ray allows: `@ray.remote(num_gpus=0.25)`.
Ray will pack 4 such actors onto 1 physical GPU.
**Use Case:** Inference servers sharing a V100.

### 3. Placement Groups (Gang Scheduling)
For Distributed Training (DDP), you need 8 workers to start **simultaneously**. If you only have 7 GPUs free, starting 7 workers is deadlock (they wait for the 8th).
**Placement Group:** Reserves a bundle of resources atomically.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: GPU Requests

#### 📁 `src/04_resources.py`
```python
import ray
import os

# Simulate a machine with 2 GPUs
ray.init(num_gpus=2)

@ray.remote(num_gpus=0.5)
def gpu_task(idx):
    gpu_ids = os.environ.get("CUDA_VISIBLE_DEVICES")
    return f"Task {idx} sees GPU: {gpu_ids}"

# Launch 4 tasks (0.5 * 4 = 2.0 GPUs)
futures = [gpu_task.remote(i) for i in range(4)]
print(ray.get(futures))
# Output Example:
# ['Task 0 sees GPU: 0', 'Task 1 sees GPU: 0', 
#  'Task 2 sees GPU: 1', 'Task 3 sees GPU: 1']
```

### 👨‍💻 Core Implementation: Placement Groups

```python
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

ray.shutdown()
ray.init(num_cpus=8)

# 1. Create a Strategy (I need 2 bundles of 2 CPUs each)
# Strategy="PACK" means put them on same node if possible
pg = placement_group([{"CPU": 2}, {"CPU": 2}], strategy="PACK")

# 2. Wait for it to be ready (Reserves the resources)
ray.get(pg.ready())

@ray.remote(num_cpus=2)
def worker():
    return "I am running inside the placement group!"

# 3. Schedule task into the group
f1 = worker.options(
    scheduling_strategy=PlacementGroupSchedulingStrategy(
        placement_group=pg,
        placement_group_bundle_index=0
    )
).remote()

print(ray.get(f1))
```

---

## 🔬 Lab Exercise: "Resource Deadlock"

### Task
Simulate a deadlock.
1.  Init Ray with `num_cpus=2`.
2.  Define Actor A that requires 1 CPU.
3.  Inside Actor A, define Actor B that requires 1 CPU and wait for it.
4.  Launch Actor A (Takes 1 CPU).
5.  Launch Actor A again (Takes 1 CPU). Total 2/2 used.
6.  Actor A tries to launch B. Pending (No CPUs).
7.  **Result:** Deadlock.
8.  **Fix:** Don't put blocking calls inside Actors unless you have spare capacity.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Gang Scheduling:** Always use Placement Groups for Distributed Training. It prevents "Partial Allocation" where resources are held but useless.
2.  **Custom Resources:** You can pass `resources={"node_role:training": 1}` to `ray.init` and request it in tasks to force code to run on specific nodes.
3.  **Head Node:** By default, the Head Node can run tasks (`num_cpus > 0`). In production, set Head Node `num_cpus=0` so it only runs the Scheduler/GCS (Global Control Store).

### API Summary
```python
ray.available_resources()
ray.cluster_resources()
```

---

**Day 87 Complete** ✅

*Next: Day 88 - Parameter Server Pattern - Implementing a distributed Training State.*

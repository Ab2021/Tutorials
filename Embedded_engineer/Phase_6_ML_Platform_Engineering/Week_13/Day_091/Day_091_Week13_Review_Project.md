# Day 91: Week 13 Review & Project - Distributed MapReduce Engine
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** Combine Tasks, Actors, and the Object Store to build a classic distributed system: A MapReduce engine capable of processing massive log files in parallel.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Architect** a MapReduce pipeline using Ray.
2.  **Implement** a `Mapper` (Task) to tokenize text.
3.  **Implement** a `Reducer` (Actor) to aggregate state.
4.  **Handle** failures using `max_retries`.
5.  **Visualize** the execution DAG.

---

## 📚 Week 13 Recap

| Day | Topic | The "Ah-ha" Moment |
|-----|-------|---------------------|
| 85 | Ray Core | "`@ray.remote` turns my function into a microservice." |
| 86 | Object Store | "Passing by Reference is instantaneous." |
| 87 | Scheduling | "I can ask for 0.5 GPUs." |
| 88 | Parameter Server | "Workers pull weights, compute gradients, push updates." |
| 89 | Resilience | "If a node dies, Ray replays the task lineage." |
| 90 | Runtime Env | "I can ship my local code to the cluster automatically." |

---

## 🏗️ Final Project: "RayReduce"

### Scenario
We have 100 log files (Total 10GB). We want to count the occurrence of the word "ERROR".
**Input:** List of file paths.
**Output:** Total count.

### Architecture
1.  **Master (Driver):** Splits input list into chunks.
2.  **Mappers (Tasks):** Read file chunk, count "ERROR", return partial count.
3.  **Reducer (Actor/Task):** Sums partial counts.

### Step 1: Data Generation

```python
# src/generate_data.py
import os
def create_logs():
    os.makedirs("logs", exist_ok=True)
    for i in range(20):
        with open(f"logs/log_{i}.txt", "w") as f:
            content = "INFO process started\n" * 10000
            content += "ERROR null pointer exception\n" * (i * 10) # Variable errors
            f.write(content)
```

### Step 2: The MapReduce Implementation

#### 📁 `project/ray_reduce.py`
```python
import ray
import os
import glob
import time

ray.init()

# MAP PHASE: Stateless Task
# Retry up to 3 times if file read fails
@ray.remote(max_retries=3)
def map_log_file(filepath):
    count = 0
    try:
        with open(filepath, "r") as f:
            for line in f:
                if "ERROR" in line:
                    count += 1
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        raise e 
    return count

# REDUCE PHASE: Smart Accumulation
@ray.remote
def reduce_counts(counts):
    return sum(counts)

def run_job():
    # 1. Get Inputs
    files = glob.glob("logs/*.txt")
    print(f"Processing {len(files)} files...")
    
    start = time.time()
    
    # 2. Launch Maps (Fan Out)
    # This creates 20 tasks in parallel
    map_refs = [map_log_file.remote(f) for f in files]
    
    # 3. Wait/Monitor (Optional)
    # We can inspect progress using ray.wait logic
    
    # 4. Launch Reduce (Fan In)
    # Note: We pass the LIST of references. 
    # The reduce task will wait for all maps to complete.
    total_ref = reduce_counts.remote(map_refs)
    
    # 5. Get Result
    total = ray.get(total_ref)
    
    duration = time.time() - start
    print(f"Total ERRORs: {total}")
    print(f"Time Taken: {duration:.4f}s")

if __name__ == "__main__":
    # Ensure data exists
    if not os.path.exists("logs"):
        print("Run generate_data.py first!")
    else:
        run_job()
```

### Step 3: Advanced Feature - The Progress Tracker Actor

Let's add an Actor to track progress in real-time.

```python
@ray.remote
class ProgressTracker:
    def __init__(self, total):
        self.total = total
        self.done = 0
    
    def task_done(self):
        self.done += 1
        print(f"Progress: {self.done}/{self.total} ({(self.done/self.total)*100:.1f}%)")

# Modified Mapper
@ray.remote
def map_with_tracking(filepath, tracker):
    # ... logic ...
    tracker.task_done.remote() # Async call to actor
    return count
```

---

## 🔬 Lab Exercise: "Verification"

### Task
1.  Generate files.
2.  Run `ray_reduce.py`.
3.  Calculated expected count (Sum of `i*10` for i=0..19). `Sum(0..19) * 10 = 190 * 10 = 1900`.
4.  Verify output matches 1900.

---

## 📝 Success Criteria
1.  **Parallelism:** User sees all CPUs utilized during Map phase (Dashboard).
2.  **Lineage:** If a Map task is killed manually, it auto-restarts.
3.  **Correctness:** Result is deterministic.

---

**Week 13 Complete** ✅

*Next Week: Week 14 - KubeRay & Production Ray - Running this on Kubernetes.*

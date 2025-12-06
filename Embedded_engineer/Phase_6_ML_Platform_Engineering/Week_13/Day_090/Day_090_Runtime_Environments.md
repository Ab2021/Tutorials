# Day 90: Dependency Hell Solved: Runtime Environments
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 13: Distributed Computing with Ray

---

> **🎯 Focus Area:** Your laptop has `code.py`. The cluster notes do not. Your laptop has `pandas`. The cluster does not. Use **Runtime Environments** to dynamically ship code and dependencies.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Configure** `runtime_env` to automatically zip and upload the local directory.
2.  **Specify** per-task Pip dependencies (creating ephemeral virtualenvs on workers).
3.  **Inject** Environment Variables (`AWS_ACCESS_KEY`) securely.
4.  **Isolate** conflicting dependencies (Task A uses Pandas 1.0, Task B uses Pandas 2.0).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- Local Machine.

### Software Environment
- `ray`.

---

## 📖 Theoretical Foundation

### 1. The Deployment Gap
Old way:
1.  Change code.
2.  Build Docker Image.
3.  Push to Registry.
4.  Restart Cluster.
5.  Run. (Slow iteration loop).

### 2. The `working_dir` Solution
Ray Solution:
1.  `ray.init(runtime_env={"working_dir": "."})`.
2.  Ray zips current folder.
3.  Uploads to GCS (Head Node).
4.  Workers download and unzip to `/tmp/...`.
5.  Set `PYTHONPATH`.
6.  Run. (Fast iteration loop).

### 3. Per-Task Virtualenvs
Ray can create a conda env or venv *on the fly* for a specific actor.
Useful for multi-tenant clusters or A/B testing different library versions.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Shipping Code

Assume we have a file `src/helper.py` that we need on the worker.

#### 📁 `src/helper.py`
```python
def secret_algorithm(x):
    return x * 42
```

#### 📁 `src/07_runtime_env.py`
```python
import ray
import os

# 1. Zip this directory and send to workers
# 2. Install 'requests' on workers
runtime_env = {
    "working_dir": "./src",
    "pip": ["requests", "pyjokes"],
    "env_vars": {"MY_SECRET": "top_secret"}
}

ray.init(runtime_env=runtime_env)

@ray.remote
def worker_task():
    # Test Import from working_dir
    from helper import secret_algorithm
    val = secret_algorithm(2)
    
    # Test Pip Package
    import pyjokes
    joke = pyjokes.get_joke()
    
    # Test Env Var
    secret = os.environ["MY_SECRET"]
    
    return f"Val: {val}, Secret: {secret}, Joke: {joke}"

print(ray.get(worker_task.remote()))
```

### 👨‍💻 Core Implementation: Conflicting Versions

```python
@ray.remote(runtime_env={"pip": ["numpy==1.21.0"]})
def old_task():
    import numpy
    return numpy.__version__

@ray.remote(runtime_env={"pip": ["numpy==1.24.0"]})
def new_task():
    import numpy
    return numpy.__version__

print(ray.get([old_task.remote(), new_task.remote()]))
# Output: ['1.21.0', '1.24.0'] running simultaneously!
```

---

## 🔬 Lab Exercise: "The Excluded File"

### Task
Manage upload size.
1.  Create a 1GB dummy file in `./src/big_data.csv`.
2.  Run the script.
3.  **Observation:** Startup is slow (Uploading 1GB).
4.  Create `.rayignore` (like gitignore). Add `*.csv`.
5.  Run again.
6.  **Observation:** Fast startup.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Fast Iteration:** `working_dir` is perfect for development. For production, baking code into Docker images (Day 77) is still safer/more reproducible.
2.  **Overhead:** Creating pip environments takes time (minutes). Use this for long-running Actors, not sub-second Tasks.
3.  **Cross-Platform:** `working_dir` works best if Driver and Worker are same OS (Linux/Linux).

### API Summary
```python
ray.init(runtime_env=...)
task.options(runtime_env=...).remote()
```

---

**Day 90 Complete** ✅

*Next: Day 91 - Week 13 Review & Project - Building a Distributed MapReduce Engine.*

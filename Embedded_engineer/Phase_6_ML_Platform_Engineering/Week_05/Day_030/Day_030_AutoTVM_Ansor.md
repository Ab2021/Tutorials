# Day 30: Automated Kernel Optimization with AutoTVM & Ansor
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 5: Deep Learning Compiler Stack

---

> **🎯 Focus Area:** Stop hand-tuning Loop Split factors. Use **AutoTVM** and **Ansor (AutoScheduler)** to automatically search the parameter space and generate state-of-the-art kernels for your specific hardware.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between AutoTVM (Template-based) and Ansor (Search-based).
2.  **Define** a Search Space using `ConfigSpace` in AutoTVM.
3.  **Run** a tuning session with an XGBoost Tuner.
4.  **Apply** best tuning records to compile a model.
5.  **Achieve** cuBLAS-level performance using pure Python definitions.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (The tuner needs to measure real execution time).

### Software Environment
```bash
pip install xgboost tornado cloudpickle
```

### Prior Knowledge
- Day 29: TVM TE and Scheduling (Split/Bind/Reorder).
- Gradient Boosting (conceptually, as the search algorithm).

---

## 📖 Theoretical Foundation

### 1. The Search Space Problem

In Day 29, we manually chose `factor=32`. Why not 16? Why not 64? Why not `split(axis, factor=128)` then `split(inner, factor=4)`?
*   **Search Space:** The combinatorial explosion of all possible valid schedules.
*   **Cost Model:** A machine learning model (XGBoost) that predicts "How fast will this schedule run?" based on features (Memory access patterns, Instruction mix).

### 2. AutoTVM vs Ansor

*   **AutoTVM:** You write a template ("I want to tile this loop, but I don't know the size"). You define knobs (`cfg.define_knob`).
    *   *Pro:* Predictable structure.
    *   *Con:* You must write the template.
*   **Ansor (AutoScheduler):** You just give the Math Compute. Ansor generates the schedule structure *and* parameters automatically using evolutionary search.
    *   *Pro:* Less code. Often faster.
    *   *Con:* Tuning takes longer (hours).

---

## 💻 Implementation

### 👨‍💻 Core Implementation: AutoTVM (Template Tuning)

We will tune the MatMul from Day 29.

#### 📁 `src/autotvm_matmul.py`
```python
#!/usr/bin/env python3
"""
Day 30: AutoTVM Matrix Multiplication Tuning
Phase 6: DL Compiler Stack
"""

import tvm
from tvm import te, autotvm
import numpy as np

# 1. Define the Task with Decorator
@autotvm.template("tutorial/matmul")
def matmul(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    
    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    
    # 2. Define the Search Space (The "Knobs")
    cfg = autotvm.get_config()
    
    # Define split candidates
    # 'tile_y' will be a number chosen from [1, 2, 4, ... 128]
    cfg.define_split("tile_y", N, num_outputs=2)
    cfg.define_split("tile_x", M, num_outputs=2)
    
    # 3. Create Schedule using Knobs
    s = te.create_schedule(C.op)
    
    # Access the chosen config
    i, j = s[C].op.axis
    
    # Apply the split determined by the tuner
    y_outer, y_inner = cfg["tile_y"].apply(s, C, i)
    x_outer, x_inner = cfg["tile_x"].apply(s, C, j)
    
    # Standard Bindings
    s[C].bind(y_outer, te.thread_axis("blockIdx.y"))
    s[C].bind(x_outer, te.thread_axis("blockIdx.x"))
    s[C].bind(y_inner, te.thread_axis("threadIdx.y"))
    s[C].bind(x_inner, te.thread_axis("threadIdx.x"))
    
    return s, [A, B, C]

def tune_and_run():
    # Setup
    N, L, M = 512, 512, 512
    task = autotvm.task.create("tutorial/matmul", args=(N, L, M, "float32"), target="cuda")
    
    print(task.config_space)
    
    # 4. Tuner
    measure_option = autotvm.measure_option(
        builder=autotvm.LocalBuilder(),
        runner=autotvm.LocalRunner(number=5, repeat=3, min_repeat_ms=100)
    )
    
    tuner = autotvm.tuner.XGBTuner(task)
    
    print("Starting Tuning (This takes time)...")
    # n_trial=20 for demo. In production, use 1000+.
    tuner.tune(
        n_trial=20,
        measure_option=measure_option,
        callbacks=[autotvm.callback.log_to_file("matmul.log")]
    )
    
    # 5. Apply History
    print("\nCompiling with Best Config...")
    with autotvm.apply_history_best("matmul.log"):
        with tvm.target.Target("cuda"):
            s, arg_bufs = matmul(N, L, M, "float32")
            func = tvm.build(s, arg_bufs)
            
    # Evaluation
    dev = tvm.cuda()
    a_np = np.random.uniform(size=(N, L)).astype(np.float32)
    b_np = np.random.uniform(size=(L, M)).astype(np.float32)
    c_np = np.zeros((N, M), dtype=np.float32)
    
    a = tvm.nd.array(a_np, dev)
    b = tvm.nd.array(b_np, dev)
    c = tvm.nd.array(c_np, dev)
    
    func(a, b, c)
    evaluator = func.time_evaluator(func.entry_name, dev, number=10)
    print(f"Optimized Time: {evaluator(a, b, c).mean * 1e3:.4f} ms")

if __name__ == "__main__":
    tune_and_run()
```

### 👨‍💻 Advanced: Ansor (AutoScheduler)

Ansor is the successor. It requires **no template**.

#### 📁 `src/ansor_demo.py`
```python
#!/usr/bin/env python3
"""
Day 30: Ansor (AutoScheduler)
"""

import tvm
from tvm import te, auto_scheduler
import numpy as np

# 1. Define Compute ONLY (No Schedule Logic!)
@auto_scheduler.register_workload
def matmul_ansor(N, L, M, dtype):
    A = te.placeholder((N, L), name='A', dtype=dtype)
    B = te.placeholder((L, M), name='B', dtype=dtype)
    k = te.reduce_axis((0, L), name='k')
    C = te.compute((N, M), lambda i, j: te.sum(A[i, k] * B[k, j], axis=k), name='C')
    return [A, B, C]

def tune_ansor():
    target = tvm.target.Target("cuda")
    N, L, M = 512, 512, 512
    
    # 2. Extract Task
    task = auto_scheduler.SearchTask(
        func=matmul_ansor, args=(N, L, M, "float32"), target=target
    )
    
    # 3. Tune
    log_file = "ansor_matmul.json"
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=20, # Low for demo
        runner=auto_scheduler.LocalRunner(repeat=1, enable_cpu_cache_flush=True),
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )
    
    task.tune(tune_option)
    
    # 4. Compile
    sch, args = task.apply_best(log_file)
    func = tvm.build(sch, args, target)
    
    print("Ansor Build Complete.")
    # (Execution code same as above)

if __name__ == "__main__":
    tune_ansor()
```

---

## 🔬 Lab Exercise: "Tuning a Neural Network"

### Lab Objectives
1.  Import a ResNet18 from Torch/Relay.
2.  Extract tasks (Ansor will find all 20 Conv layers).
3.  Tune the network (This runs for hours in real life, do 10 trials for lab).
4.  Observe speedup vs unoptimized baseline.

**Concept:**
Compilers treat improved performance as a search problem. You trade *Compile Time* (Search) for *Runtime* (Inference) speed.

---

## 📝 Daily Summary

### Key Takeaways
1.  **Knobs:** Optimization parameters (Tile sizes, Unroll factors, Vectorization widths) that don't change correctness but drastically affect speed.
2.  **Machine Learning for Machine Learning:** We use XGBoost (in AutoTVM) to predict the speed of a customized CUDA kernel, saving us from running every single candidate on the GPU.
3.  **Ansor is Powerful:** It can discover strategies (compute at root, cache read, cooperative fetch) that average humans might miss.

### API Summary
```python
# AutoTVM
cfg.define_split("tile_k", k, num_outputs=2)
tuner = autotvm.tuner.XGBTuner(task)

# Ansor
task = auto_scheduler.SearchTask(...)
task.tune(tune_options)
```

---

**Day 30 Complete** ✅

*Next: Day 31 - MLIR (Multi-Level Intermediate Representation) - The foundation of modern compilers (and TensorFlow/JAX).*

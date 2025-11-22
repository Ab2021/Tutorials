# Day 1: Modern AI Engineering Landscape
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Explain the difference between pip and Poetry for dependency management. Why would you choose one over the other for an ML project?

**Answer:**

**pip with requirements.txt:**
- Simple, widely used
- `requirements.txt` specifies package names and optional version constraints
- Problem: **No dependency resolution**. If package A needs library@>=1.0 and package B needs library@<1.0, pip might install incompatibly
- No lock file by default (can use `pip freeze`, but captures current environment, not intent)

**Poetry:**
- Modern dependency management with SAT solver
- **Resolves dependencies**: Finds compatible versions satisfying all constraints
- `pyproject.toml`: Declares direct dependencies with constraints
- `poetry.lock`: Records exact versions and hashes of all dependencies (transitive too)
- **Reproducibility**: `poetry install` on any machine gives identical environment

**When to choose:**
- **pip**: Simple scripts, quick prototypes, wide compatibility needed
- **Poetry**: Production ML systems where reproducibility is critical, complex dependency trees, team collaboration

**ML-specific considerations:**
- ML  libraries (torch, transformers, tensorflow) have complex dependencies (CUDA versions, etc.)
- Poetry's resolution prevents version conflicts
- Lock files ensure training today produces same results as tomorrow

---

#### Q2: What is torch.compile in PyTorch 2.0, and when would you NOT want to use it?

**Answer:**

**What it does:**
- JIT (Just-In-Time) compiles PyTorch models
- Captures computation graph during first run (TorchDynamo)
- Optimizes graph (kernel fusion, constant folding, layout optimization)
- Generates optimized code (TorchInductor)
- Typical speedup: 30-50%

**When NOT to use:**

1. **Dynamic Shapes**: If your model has highly dynamic input shapes (varying batch sizes, sequence lengths), torch.compile will recompile for each new shape. Overhead > speedup.

   ```python
   # BAD for torch.compile
   for batch in dataloader:
       # batch.shape changes every iteration: (2, 50), (8, 120), (4, 75)
       output = model(batch)
   ```

2. **Rapid Development**: Compilation overhead on first run (seconds to minutes). During hyperparameter tuning where you restart frequently, this overhead accumulates.

3. **Unsupported Operations**: Some ops don't compile yet (certain custom CUDA kernels, some third-party libraries). Falls back to eager mode but with overhead.

4. **Debugging**: Compiled models are harder to debug (can't step through line-by-line). During development, eager mode is clearer.

**Best practice:**
- Dev: Eager mode for debugging
- Production: torch.compile for inference (shapes usually static)
- Training: Compile if training loop is stable and shapes are fixed

---

#### Q3: You're training a 13B parameter model and getting "CUDA out of memory" errors on an 80GB A100. What are your debugging and mitigation strategies?

**Answer:**

**Debugging Steps:**

1. **Calculate Expected Memory:**

   For AdamW optimizer in FP16:
   - Model weights: 13B × 2 bytes = 26 GB
   - Gradients: 13B × 2 bytes = 26 GB  
   - Optimizer states (momentum + variance): 13B × 2 × 4 bytes = 104 GB
   - **Total**: ~156 GB > 80 GB ← Problem identified!

2. **Check Actual Usage:**
   ```python
   import torch
   print(torch.cuda.memory_summary())
   # Shows: allocated, reserved, fragmentation
   ```

**Mitigation Strategies (in order of impact):**

1. **Reduce Batch Size** (Most Immediate):
   - Batch size 8 → 4 (halves activation memory)
   - Trade-off: Longer training time

2. **Gradient Accumulation** (Maintain Effective Batch Size):
   ```python
   optimizer.zero_grad()
   for i, batch in enumerate(dataloader):
       loss = model(batch) / accumulation_steps
       loss.backward()
       if (i + 1) % accumulation_steps == 0:
           optimizer.step()
           optimizer.zero_grad()
   ```
   - Accumulate 4 steps: same gradient as batch=32, memory of batch=8

3. **Mixed Precision** (2× savings):
   ```python
   with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
       loss = model(batch)
   ```
   - Weights and activations in BF16 (2 bytes vs 4 bytes)

4. **Gradient Checkpointing** (Trade Compute for Memory):
   ```python
   from torch.utils.checkpoint import checkpoint
   # Don't store intermediate activations, recompute on backward pass
   output = checkpoint(layer, input)
   ```
   - Can save 30-50% activation memory
   - Increases training time by ~20%

5. **Optimizer State Sharding (FSDP/DeepSpeed)**:
   If multi-GPU available:
   ```python
   from torch.distributed.fsdp import FullyShardedDataParallel
   model = FSDP(model, sharding_strategy=FULL_SHARD)
   ```
   - Shards optimizer states across GPUs

6. **8-bit Optimizers**:
   ```python
   import bitsandbytes as bnb
   optimizer = bnb.optim.AdamW8bit(model.parameters())
   ```
   - Optimizer states in 8-bit: 104 GB → 26 GB

**Final Answer for 13B on A100:**
- Mixed precision (BF16): 26 + 26 + 52 (8-bit optimizer) = 104 GB → Still doesn't fit!
- Add gradient checkpointing: ~70 GB ✓ Fits!

---

#### Q4: Explain FSDP (Fully Sharded Data Parallel). How does it differ from standard Data Parallel (DDP)?

**Answer:**

**Data Parallel (DDP):**
- Each GPU has complete copy of model
- Forward pass: Each GPU processes different batch
- Backward pass: Gradients all-reduced (averaged across GPUs)
- Optimizer step: Each GPU updates its model copy
- **Memory**: Full model per GPU
- **Communication**: One all-reduce per iteration (gradients only)

**FSDP (Fully Sharded Data Parallel):**
- Model parameters sharded across GPUs (each GPU owns 1/N of parameters)
- Forward pass:
  1. All-gather parameters for current layer
  2. Compute forward
  3. Discard gathered params (keep only local shard)
- Backward pass: Similar (all-gather, compute, discard)
- Optimizer step: Each GPU updates only its local shard
- **Memory**: 1/N model per GPU (N = number of GPUs)
- **Communication**: All-gather per layer (more frequent, but smaller)

**Comparison:**

| Aspect | DDP | FSDP |
|--------|-----|------|
| Model memory | Full model/GPU | Model/N GPUs |
| Optimizer memory | Full optimizer/GPU | Optimizer/N GPUs |
| Communication | 1× per iteration (all-reduce) | 2N× per iteration (all-gather forward+backward) |
| When to use | Model fits in GPU | Model doesn't fit in single GPU |

**Trade-off:**
- FSDP: Lower memory, higher communication
- Works when: Communication time < Compute time (true for large models)

**Interview Follow-up:**
*Q: What are the downsides of FSDP?*

**A:** 
1. Communication overhead (especially with slow interconnects)
2. Complexity: Harder to debug than DDP
3. Not all operations supported (some require full model)
4. Prefetching and overlap crucial for performance (implementation complexity)

---

#### Q5: Your ML pipeline suddenly starts producing different results (accuracy drops from 85% to 82%), but no code changed. How do you debug this?

**Answer:**

This is a **reproducibility failure**. Systematic debugging:

**1. Check Data Drift:**
```python
# Compare current data distribution vs historical
import pandas as pd
current_stats = df.describe()
historical_stats = load_historical_stats()
diff = current_stats - historical_stats
# Look for significant changes in mean, std, min, max
```

Possible causes:
- Data source changed (new crawler, API update)
- Temporal shift (model trained on summer data, now winter)
- Data corruption or pipeline bug

**2. Check Library Versions:**
```python
import torch, transformers, numpy
print(f"PyTorch: {torch.__version__}")
print(f"Transformers: {transformers.__version__}")
print(f"NumPy: {numpy.__version__}")
# Compare with versions used in training
```

Possible causes:
- Auto-upgrade of dependencies (if not using lock files!)
- transformers library behavior change (default tokenization, attention masks)
- PyTorch optimization changes

**3. Check Model Weights:**
```python
import hashlib
import torch

model = load_model()
weights_bytes = torch.save(model.state_dict(), BytesIO())
current_hash = hashlib.sha256(weights_bytes).hexdigest()
print(f"Model hash: {current_hash}")
# Compare with production model hash
```

Possible causes:
- Wrong model checkpoint loaded
- Partial checkpoint corruption
- Float precision differences (FP32 vs FP16 loading)

**4. Check Input Processing:**
```python
# Log inputs before model
print(f"Input shape: {input_tensor.shape}")
print(f"Input dtype: {input_tensor.dtype}")
print(f"Input stats: mean={input_tensor.mean()}, std={input_tensor.std()}")
print(f"Sample values: {input_tensor[0, :10]}")
```

Possible causes:
- Tokenization changes (vocabulary, special tokens)
- Normalization changes (mean/std updated)
- Data augmentation randomness (if not seeded)

**5. Check Hardware/CUDA:**
```python
print(f"CUDA version: {torch.version.cuda}")
print(f"cuDNN version: {torch.backends.cudnn.version()}")
print(f"Device: {torch.cuda.get_device_name()}")
```

Possible causes:
- CUDA non-determinism (can use `torch.use_deterministic_algorithms(True)`)
- GPU architecture change (V100 → A100, different precision behavior)

**Systematic Approach:**
1. Isolate: Run model on known good data (test set from training time)
2. Version control: Check git commit, Docker image tag
3. A/B test: Run old version and new version side-by-side
4. Binary search: If code history changed, binary search good → bad commit

**Prevention:**
- Pin all dependencies (Poetry/lock files)
- Version control data (DVC, hash datasets)
- Log model hashes and versions
- Use Docker for complete environment reproducibility
- Implement regression tests (fixed inputs → expected outputs)

---

### Production Challenges

#### Challenge 1: Dependency Hell in Production

**Scenario:**
Your model works perfectly locally but fails in production with:
```
ImportError: cannot import name 'AutoModelForCausalLM' from 'transformers'
```

Local: `transformers==4.35.2`
Production: `transformers==4.20.1` (old requirements.txt not updated)

**Root Cause:**
- Requirements.txt had `transformers>=4.20.0`
- Local pip install picked latest (4.35.2)  
- Production had cached 4.20.1 and didn't upgrade
- `AutoModelForCausalLM` added in 4.25.0

**Solution:**
1. Immediate: Pin exact version `transformers==4.35.2`
2. Long-term: Use Poetry with lock files
3. Testing: CI/CD should test on production-like environment
4. Docker: Build once, run anywhere

---

#### Challenge 2: CUDA Version Mismatch

**Scenario:**
Model trained locally loads with errors:
```
RuntimeError: CUDA error: no kernel image is available for execution on the device
```

Local: CUDA 12.1, PyTorch compiled for cu121
Production: CUDA 11.8

**Root Cause:**
PyTorch binaries are compiled for specific CUDA versions. cu121 binaries don't run on CUDA 11.8.

**Solution:**
```dockerfile
# Dockerfile with explicit CUDA version
FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

# Install PyTorch for CUDA 12.1
RUN pip install torch==2.1.0+cu121 -f https://download.pytorch.org/whl/torch_stable.html
```

**Prevention:**
- Document CUDA requirements
- Use containers with bundled CUDA
- Or: Use CPU-only for inference (quantized models)

---

#### Challenge 3: Model Versioning Chaos

**Scenario:**
Data scientist: "Our model achieves 92% accuracy!"
Three months later: "I can't reproduce the 92% result, getting 88%"

**Root Causes:**
- Model checkpoint not versioned (overwritten)
- Dataset changed (new data added)
- Code changed (preprocessing logic)
- Random seed not set

**Solution: MLflow/Weights & Biases Tracking**

```python
import mlflow

mlflow.start_run()
mlflow.log_param("model_name", "llama-2-7b")
mlflow.log_param("learning_rate", 2e-5)
mlflow.log_param("batch_size", 16)
mlflow.log_param("dataset_hash", compute_dataset_hash())
mlflow.log_param("code_commit", get_git_commit())

# Training...
mlflow.log_metric("accuracy", accuracy)
mlflow.pytorch.log_model(model, "model")
mlflow.end_run()
```

Now every experiment is tracked with:
- Hyperparameters
- Dataset version
- Code commit
- Resulting metrics
- Model artifacts

**Reproduction:**
```python
# Load exact experiment configuration
run = mlflow.get_run(run_id="abc123")
params = run.data.params
# Checkout code commit, load dataset version, retrain
```

---

### Key Takeaways for Interviews

1. **Understand Trade-offs**: Every tool has costs (memory, compute, complexity)
2. **Debugging Methodology**: Systematic isolation, not random changes
3. **Production Thinking**: Reproducibility, monitoring, versioning
4. **Communication**: Explain WHY, not just WHAT you would do
5. **Real-world Experience**: Mention specific tools (Poetry, FSDP, MLflow) shows hands-on knowledge

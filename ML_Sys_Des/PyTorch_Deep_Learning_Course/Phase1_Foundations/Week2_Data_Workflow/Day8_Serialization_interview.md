# Day 8: Serialization - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Deployment, Security, and Formats

### 1. Why is `pickle` considered insecure?
**Answer:**
*   Pickle is a stack-based virtual machine.
*   It can construct arbitrary Python objects and execute arbitrary code (e.g., `os.system('rm -rf /')`) during the unpickling process.
*   Never unpickle untrusted data. Use `safetensors` or ONNX.

### 2. What is the difference between `torch.save(model)` and `torch.save(model.state_dict())`?
**Answer:**
*   `save(model)`: Saves the entire object using pickle. Requires the class definition to be available when loading. Fragile to code refactoring.
*   `save(state_dict)`: Saves only the parameters (Key-Value pairs). Recommended. Agnostic to class structure as long as keys match.

### 3. How do you load a model trained on GPU onto a CPU machine?
**Answer:**
*   `torch.load('model.pth', map_location='cpu')`.
*   Without `map_location`, PyTorch tries to restore tensors to the device they were saved from (GPU), which fails if no GPU is present.

### 4. What is `strict=False` in `load_state_dict`?
**Answer:**
*   It ignores non-matching keys.
*   **Missing Keys**: Layers in model but not in state_dict (Randomly initialized).
*   **Unexpected Keys**: Layers in state_dict but not in model (Ignored).
*   Essential for Transfer Learning / Fine-tuning.

### 5. Explain ONNX.
**Answer:**
*   Open Neural Network Exchange.
*   A graph representation of the model using standard operators.
*   Allows training in PyTorch and deploying in C++ (ONNX Runtime), Java, or converting to TensorRT.

### 6. What is `safetensors`?
**Answer:**
*   A secure, fast, zero-copy serialization format developed by Hugging Face.
*   Does not use pickle.
*   Uses memory mapping for fast loading of large models (LLMs).

### 7. How do you resume an interrupted training session?
**Answer:**
*   You need to save and load:
    1.  Model weights.
    2.  Optimizer state (momentum buffers).
    3.  Scheduler state (current LR).
    4.  Epoch number / Step count.
    5.  Random seeds (optional, for strict reproducibility).

### 8. What happens to the Optimizer state when you change the Learning Rate manually?
**Answer:**
*   The optimizer state (momentum) is preserved.
*   Only the `param_groups['lr']` changes.
*   This is generally safe and common (LR scheduling).

### 9. Can you load weights from a different architecture?
**Answer:**
*   Only if the tensor shapes match.
*   You can manually manipulate the dictionary keys/shapes before loading to make them fit (e.g., "Surgery" on the state dict).

### 10. What is "Model Quantization" in the context of saving?
**Answer:**
*   Saving weights as `int8` instead of `float32`.
*   Reduces file size by 4x.
*   Requires quantization-aware training or post-training quantization to maintain accuracy.

### 11. How does `torch.jit.trace` work?
**Answer:**
*   Runs the model with a dummy input.
*   Records the sequence of operations executed.
*   Produces a static graph.
*   Fails to capture dynamic control flow (`if x > 0`).

### 12. How does `torch.jit.script` work?
**Answer:**
*   Analyzes the Python source code (AST).
*   Compiles it into TorchScript intermediate representation.
*   Supports control flow.
*   Requires code to be written in a specific subset of Python (TorchScript compatible).

### 13. What is a "State Dict"?
**Answer:**
*   An `OrderedDict` mapping parameter names to tensors.
*   Includes both Parameters (`requires_grad=True`) and Buffers (`running_mean`).

### 14. How do you save a model to S3 or GCS directly?
**Answer:**
*   `torch.save` accepts a file-like object.
*   Open a stream to S3 (`io.BytesIO`) and save to it.
*   Or use libraries like `smart_open`.

### 15. What is "Lazy Loading" of modules?
**Answer:**
*   Defining layers on the `meta` device (no memory allocated).
*   Loading weights materializes them.
*   Useful for loading 100B+ models on limited RAM.

### 16. Why might `load_state_dict` fail even if keys match?
**Answer:**
*   **Shape Mismatch**: The tensor size in file differs from model.
*   **Dtype Mismatch**: Trying to load float16 into float32 (usually auto-casted, but can error).

### 17. How do you handle versioning of model checkpoints?
**Answer:**
*   Include metadata in the checkpoint dictionary: `{'version': '1.0', 'git_hash': '...', ...}`.
*   Use tools like MLflow or DVC.

### 18. What is `torch.package` (Digress)?
**Answer:**
*   A format to bundle model code + weights + dependencies into a single file.
*   Makes models self-contained.

### 19. How do you debug a "KeyError" during loading?
**Answer:**
*   Print `model.state_dict().keys()` and `loaded_dict.keys()`.
*   Check for prefixes like `module.` (added by `DataParallel`) or `_orig_mod.` (added by `torch.compile`).

### 20. What is the benefit of Zero-Copy loading?
**Answer:**
*   Using `mmap` (in safetensors).
*   The OS maps the file directly to RAM.
*   Instant loading. No CPU copy loop.
*   Allows loading models larger than RAM (OS handles paging).

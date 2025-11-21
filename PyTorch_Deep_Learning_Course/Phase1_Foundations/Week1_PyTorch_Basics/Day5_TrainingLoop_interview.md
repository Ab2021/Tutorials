# Day 5: The Training Loop - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: Training Pipeline, Debugging, and Convergence

### 1. Why do we need `optimizer.zero_grad()`? What happens if we forget it?
**Answer:**
*   PyTorch accumulates gradients (`+=`) in the `.grad` attribute.
*   This allows gradient accumulation across batches.
*   If we forget it, gradients from the current batch are added to gradients from the previous batch. The step direction becomes a mix of old and new, leading to divergence or erratic behavior.

### 2. What is "Gradient Scaling" in Mixed Precision Training?
**Answer:**
*   In FP16, small gradients (e.g., $10^{-5}$) underflow to zero.
*   Gradient Scaling multiplies the loss by a large factor (e.g., $2^{16}$) before backprop.
*   This shifts the gradient values into the representable range of FP16.
*   They are divided back (unscaled) before the optimizer update.

### 3. How do you detect Overfitting from Loss Curves?
**Answer:**
*   **Train Loss**: Continues to decrease.
*   **Validation Loss**: Decreases initially, then plateaus, and starts increasing.
*   The point of divergence is where overfitting starts.

### 4. What is `pin_memory=True` in DataLoader?
**Answer:**
*   It allocates the data tensors in Page-Locked (Pinned) RAM on the host.
*   Transfers from Pinned RAM to GPU are faster and can be asynchronous (overlap with compute).
*   Always use it when using a GPU.

### 5. Why is `num_workers` important?
**Answer:**
*   It spawns subprocesses to load and augment data in parallel.
*   If `num_workers=0` (default), data loading happens on the main thread, blocking the GPU training loop.
*   Setting it to 4 or 8 keeps the GPU fed.

### 6. What is the difference between `loss.item()` and `loss`?
**Answer:**
*   `loss` is a Tensor (attached to the graph). Storing it accumulates history (memory leak).
*   `loss.item()` returns the Python float value.
*   Always use `.item()` for logging/printing.

### 7. Explain "Teacher Forcing" in RNN training.
**Answer:**
*   Instead of feeding the model's own previous prediction as input for the next step, we feed the *ground truth* token.
*   Stabilizes training and converges faster.
*   Exposure Bias: Model might fail at inference time because it never learned to recover from its own mistakes.

### 8. What is "Gradient Clipping"?
**Answer:**
*   Limiting the norm of the gradient vector to a maximum value (e.g., 1.0).
*   Prevents Exploding Gradients, common in RNNs or deep networks.
*   `torch.nn.utils.clip_grad_norm_`.

### 9. How do you handle "NaN" loss?
**Answer:**
*   **Causes**: High Learning Rate, Exploding Gradients, Division by Zero, Log of Zero/Negative.
*   **Debug**: Use `torch.autograd.detect_anomaly()`. Check inputs for NaNs. Lower LR. Use Gradient Clipping.

### 10. What is the difference between `CrossEntropyLoss` and `NLLLoss`?
**Answer:**
*   `CrossEntropyLoss` = `LogSoftmax` + `NLLLoss`.
*   It expects raw logits (unnormalized scores).
*   `NLLLoss` expects Log-Probabilities (output of LogSoftmax).

### 11. Why is the Validation set necessary? Why not just Test set?
**Answer:**
*   Validation is used for **Model Selection** and **Hyperparameter Tuning**.
*   If we tune on the Test set, we overfit to the Test set (Data Leakage).
*   Test set should be used *only once* for the final report.

### 12. What is "Data Leakage"?
**Answer:**
*   Information from the test/validation set leaks into the training process.
*   Examples: Normalizing data using global mean (including test data), Time-series shuffling, Duplicate samples in train/test.

### 13. How does `torch.backends.cudnn.benchmark = True` improve performance?
**Answer:**
*   cuDNN has multiple algorithms for Convolutions (GEMM, FFT, Winograd).
*   Benchmark mode runs a quick test at the start to find the fastest algorithm for the specific input size and hardware.
*   Useful if input sizes are constant.

### 14. What is "Label Smoothing"?
**Answer:**
*   Instead of one-hot targets `[0, 1, 0]`, use soft targets `[0.1, 0.8, 0.1]`.
*   Prevents the model from becoming over-confident (logits $\to \infty$).
*   Improves generalization and calibration.

### 15. How do you implement "Accumulated Gradients"?
**Answer:**
*   Loop over micro-batches.
*   Compute loss, divide by N.
*   `loss.backward()`.
*   Step optimizer only every N steps.

### 16. What is the "Epoch" vs "Step" distinction in logging?
**Answer:**
*   **Epoch**: High-level cycle. Good for overall progress.
*   **Step**: Granular update. Good for debugging instability within an epoch.
*   TensorBoard usually plots against Steps.

### 17. Why should you shuffle the training data?
**Answer:**
*   SGD assumes data is i.i.d (Independent and Identically Distributed).
*   If data is sorted by class, the model will oscillate (learns only Class A, then forgets it and learns Class B).
*   Shuffling breaks correlation.

### 18. What is "Curriculum Learning"?
**Answer:**
*   Training on easy examples first, then gradually introducing harder ones.
*   Mimics human learning. Can speed up convergence.

### 19. How do you save a model for inference vs resuming training?
**Answer:**
*   **Inference**: Save only `model.state_dict()`.
*   **Resuming**: Save `model.state_dict()`, `optimizer.state_dict()`, `scheduler.state_dict()`, and `epoch`.

### 20. What is the impact of Batch Size on Generalization?
**Answer:**
*   Large Batch: Sharp Minima (Poor generalization).
*   Small Batch: Flat Minima (Good generalization).
*   "Linear Scaling Rule": If increasing batch size, increase LR proportionally.

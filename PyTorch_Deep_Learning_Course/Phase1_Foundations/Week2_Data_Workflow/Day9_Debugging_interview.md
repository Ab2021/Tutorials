# Day 9: Debugging - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Troubleshooting, Profiling, and Visualization

### 1. How do you debug "NaN" loss?
**Answer:**
*   **Anomaly Detection**: `torch.autograd.detect_anomaly()` to find the culprit op.
*   **Gradient Norm**: Check if gradients are exploding.
*   **Input Data**: Check for NaNs or Infs in input.
*   **Operations**: Check for `log(0)`, `sqrt(-1)`, or division by zero (epsilon).

### 2. Why is the "Overfit on One Batch" test important?
**Answer:**
*   It isolates the model capacity and optimization logic from generalization issues.
*   If a model cannot memorize a single batch, it has a bug (e.g., disconnected graph, frozen weights, wrong loss).
*   It's the "Hello World" of model debugging.

### 3. What is the difference between `torch.cuda.synchronize()` and `time.sleep()`?
**Answer:**
*   `synchronize()` blocks the CPU thread until all GPU kernels in the stream are finished. Essential for timing.
*   `sleep()` just pauses the CPU. GPU keeps running.

### 4. How do you identify a Data Loading bottleneck?
**Answer:**
*   **GPU Utilization**: Check `nvidia-smi`. If it fluctuates (sawtooth pattern) or is low (<80%), GPU is starving.
*   **Profiler**: Look for gaps in GPU timeline where "DataLoader" is running on CPU.

### 5. What are PyTorch Hooks used for?
**Answer:**
*   Injecting logic into the computation graph without modifying the class code.
*   **Forward**: Feature extraction, Activation monitoring.
*   **Backward**: Gradient clipping, Gradient monitoring.

### 6. Explain "Dying ReLU" and how to visualize it.
**Answer:**
*   Neurons outputting 0 for all inputs.
*   **Visualize**: Plot histogram of activations. If a spike at 0 is huge and persists, neurons are dead.
*   **Visualize**: Plot gradient norms. Dead neurons have 0 gradient.

### 7. What is `torch.autograd.gradcheck`?
**Answer:**
*   A utility to verify the correctness of the backward pass.
*   Compares analytical gradient (computed by `.backward()`) with numerical gradient (finite difference approximation).
*   Used when writing custom Autograd functions.

### 8. Why might Validation Loss be lower than Train Loss?
**Answer:**
*   **Dropout**: Active in Train (higher loss), inactive in Val (lower loss).
*   **Data Augmentation**: Harder examples in Train (crops/rotations), clean examples in Val.
*   **Lag**: Val loss is computed at end of epoch (weights are better), Train loss is average over epoch.

### 9. How do you visualize High-Dimensional Embeddings?
**Answer:**
*   **t-SNE** or **UMAP**.
*   Project 128D vectors to 2D/3D.
*   TensorBoard Embedding Projector supports this.

### 10. What is "Gradient Explosion" vs "Gradient Vanishing"?
**Answer:**
*   **Explosion**: Grads > 1. Weights update too much. NaN. Fix: Clipping.
*   **Vanishing**: Grads < 1. Weights don't update. Fix: Residuals, BatchNorm.

### 11. How do you measure FLOPS of a model?
**Answer:**
*   Use libraries like `fvcore` or `thop`.
*   Theoretical count of Multiply-Accumulate operations.
*   Note: FLOPS $\neq$ Latency (Memory access matters more).

### 12. What is the "saddle point" problem in visualization?
**Answer:**
*   Loss landscape visualization (2D slice) might show a minimum, but it's actually a saddle point in other dimensions.
*   Visualizations are projections and can be misleading.

### 13. How do you debug a silent memory leak in PyTorch?
**Answer:**
*   Check for lists accumulating tensors (e.g., `losses.append(loss)` instead of `loss.item()`).
*   Check for reference cycles.
*   Use `torch.cuda.memory_summary()`.

### 14. What is "Saliency Map"?
**Answer:**
*   Visualizing which pixels in the input contributed most to the prediction.
*   Compute gradient of Output Class Score w.r.t Input Image.
*   $\nabla_x y$.

### 15. Why use `tensorboard` over `print`?
**Answer:**
*   **History**: Tracks trends over time.
*   **Comparison**: Overlay multiple runs.
*   **Rich Media**: Images, Audio, Histograms.

### 16. What is "Active Learning" in the context of debugging?
**Answer:**
*   Visualizing which samples the model is most uncertain about (Entropy).
*   Helps find mislabeled data or edge cases.

### 17. How do you check if your GPU is using Tensor Cores?
**Answer:**
*   Profiler shows kernel names. Look for `s884` or `h1688` (Ampere) in kernel names.
*   Ensure inputs are FP16/BF16 and dimensions are multiples of 8.

### 18. What is "Weight Histogram" useful for?
**Answer:**
*   Checking initialization (Gaussian?).
*   Checking if weights are growing too large (need regularization).
*   Checking if weights are collapsing to 0.

### 19. How do you debug Distributed Data Parallel (DDP) hangs?
**Answer:**
*   Usually caused by rank mismatch or one process crashing.
*   Set `NCCL_DEBUG=INFO` env var to see communication logs.
*   Ensure all processes reach the same number of `optimizer.step()` calls.

### 20. What is "Class Activation Mapping" (CAM)?
**Answer:**
*   Technique to visualize heatmap of class activation.
*   Grad-CAM uses gradients flowing into the final Conv layer to weight the feature maps.

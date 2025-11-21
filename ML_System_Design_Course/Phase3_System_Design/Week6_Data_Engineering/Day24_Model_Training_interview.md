# Day 24: Model Training - Interview Questions

> **Topic**: Distributed Training
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is Data Parallelism?
**Answer:**
*   Replicate model on N GPUs.
*   Split Batch into N mini-batches.
*   Each GPU computes gradients on its chunk.
*   Average gradients (All-Reduce). Update weights.

### 2. What is Model Parallelism?
**Answer:**
*   Split Model across N GPUs (Layer 1-10 on GPU 1, 11-20 on GPU 2).
*   Necessary when model doesn't fit in VRAM.
*   **Pipeline Parallelism** helps utilization.

### 3. Explain the Parameter Server architecture.
**Answer:**
*   **Workers**: Compute gradients.
*   **Servers**: Store weights.
*   Workers push gradients to servers, servers update weights, workers pull new weights.
*   Can be Async.

### 4. Explain Ring All-Reduce.
**Answer:**
*   Bandwidth-optimal algorithm for syncing gradients.
*   Nodes pass data in a logical ring.
*   Avoids bottleneck of a central server. Used in NCCL/Horovod.

### 5. What is Synchronous vs Asynchronous SGD?
**Answer:**
*   **Sync**: Wait for all workers. Equivalent to large batch. Slowest worker is bottleneck.
*   **Async**: Workers update whenever ready. Faster but stale gradients (noise).

### 6. What is Gradient Accumulation?
**Answer:**
*   Simulate large batch size by running multiple forward/backward passes before updating weights.
*   Saves memory.

### 7. What is Checkpointing?
**Answer:**
*   Saving model state (weights + optimizer state) to disk periodically.
*   Fault tolerance (Spot instance preemption).

### 8. What is Mixed Precision Training (AMP)?
**Answer:**
*   Use FP16 for matmul, FP32 for accumulation.
*   **Loss Scaling**: Multiply loss by factor to prevent underflow in FP16 gradients.

### 9. What is ZeRO (Zero Redundancy Optimizer)?
**Answer:**
*   DeepSpeed technique.
*   Shards Optimizer State, Gradients, and Parameters across GPUs.
*   Eliminates memory redundancy in Data Parallelism.

### 10. How do you handle "Stragglers" (Slow workers)?
**Answer:**
*   **Async SGD**.
*   **Backup Workers**: Launch 11 workers, take first 10 results. Kill the slowest.

### 11. What is Hyperparameter Tuning?
**Answer:**
*   Finding optimal LR, Batch Size, etc.
*   **Grid Search**, **Random Search**, **Bayesian Optimization**.

### 12. Explain Bayesian Optimization.
**Answer:**
*   Builds a probabilistic model (Gaussian Process) of the objective function.
*   Chooses next point to evaluate by balancing Exploration (High uncertainty) and Exploitation (High predicted value).

### 13. What is Neural Architecture Search (NAS)?
**Answer:**
*   Automating network design.
*   RL or Evolutionary algorithms to find best architecture. Expensive.

### 14. What is Transfer Learning?
**Answer:**
*   Fine-tuning pretrained model.
*   Requires much less data and compute.

### 15. What is "Catastrophic Forgetting"?
**Answer:**
*   Model forgets Task A when trained on Task B.
*   **Fix**: Replay buffer, Elastic Weight Consolidation (EWC).

### 16. How do you debug a loss that doesn't decrease?
**Answer:**
*   LR too high?
*   Labels shuffled?
*   Input normalization wrong?
*   Zero initialization?

### 17. What is the "NaN Loss" problem?
**Answer:**
*   Exploding gradient or Division by zero.
*   **Fix**: Gradient Clipping, Check numerical stability (epsilon).

### 18. What is Distributed Data Parallel (DDP) in PyTorch?
**Answer:**
*   Multi-process parallelism.
*   Each GPU is a process.
*   Uses Ring All-Reduce. Recommended over DataParallel (Thread-based).

### 19. What is Horovod?
**Answer:**
*   Framework for distributed training (Uber).
*   Standardized API for TensorFlow/PyTorch.
*   Uses MPI and NCCL.

### 20. How do you estimate training time?
**Answer:**
*   Time per step $\times$ Steps per epoch $\times$ Epochs.
*   Factor in communication overhead (scaling efficiency < 100%).

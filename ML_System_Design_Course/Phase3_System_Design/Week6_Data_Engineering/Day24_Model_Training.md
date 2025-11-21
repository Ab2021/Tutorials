# Day 24: Model Training Patterns

> **Phase**: 3 - System Design
> **Week**: 6 - Data Engineering
> **Focus**: Scaling the Learning Process
> **Reading Time**: 50 mins

---

## 1. Distributed Training

When the model or data doesn't fit on one GPU.

### 1.1 Data Parallelism (DDP)
*   **Scenario**: Model fits on 1 GPU. Data is huge.
*   **Mechanism**:
    1.  Replicate model on N GPUs.
    2.  Split batch into N mini-batches.
    3.  Each GPU computes gradients on its mini-batch.
    4.  **All-Reduce**: Average gradients across all GPUs.
    5.  Update weights.

### 1.2 Model Parallelism / Pipeline Parallelism
*   **Scenario**: Model (e.g., GPT-4) is too big for 1 GPU VRAM.
*   **Mechanism**: Split the layers across GPUs. GPU 1 holds layers 1-10, GPU 2 holds 11-20.
*   **Pipeline**: While GPU 2 processes batch $i$, GPU 1 can start processing batch $i+1$ to avoid idle time.

---

## 2. Hyperparameter Tuning (HPO)

Grid Search is dead.

### 2.1 Bayesian Optimization
*   **Idea**: Build a probabilistic model (Gaussian Process) of the function `Hyperparams -> Validation Loss`.
*   **Strategy**: Intelligently choose the next set of hyperparameters to try, balancing Exploration (high uncertainty) and Exploitation (low predicted loss).

### 2.2 Early Stopping (Hyperband)
*   **Idea**: Start 100 random configs. Run for 1 epoch. Kill the bottom 50%. Run survivors for 2 epochs. Repeat.
*   **Benefit**: Don't waste resources on bad configs.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Stragglers
**Scenario**: In distributed training, one GPU is slower (thermal throttling).
**Result**: All other GPUs wait for it at the synchronization step. The whole cluster runs at the speed of the slowest node.
**Solution**:
*   **Asynchronous SGD**: Don't wait. Update weights whenever a gradient arrives. (Noisier, harder to converge).
*   **Hardware Health Checks**: Automatically drain slow nodes.

### Challenge 2: NaN Loss
**Scenario**: Halfway through a week-long training run, Loss becomes `NaN`.
**Solution**:
*   **Checkpointing**: Save model state every hour. Resume from last good checkpoint.
*   **Gradient Clipping**: Prevent explosion.
*   **Mixed Precision (FP16)**: Use `GradScaler` to prevent underflow.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Explain the "All-Reduce" operation.**
> **Answer**: It is a collective communication primitive. Every node starts with a value (gradient). At the end, every node has the *sum* (or average) of all values. Efficient implementations (Ring All-Reduce) do this in bandwidth-optimal steps.

**Q2: What is Gradient Accumulation?**
> **Answer**: A trick to simulate a large batch size when you have limited VRAM.
> *   Instead of updating weights every step, you run forward/backward passes for $N$ steps, accumulating the gradients (`grad += new_grad`).
> *   Update weights once after $N$ steps.
> *   This allows training with effective batch size 128 even if GPU can only fit 16.

**Q3: Why is Random Search often better than Grid Search?**
> **Answer**: In high dimensions, some hyperparameters matter much more than others. Grid search might waste time checking 10 values of an unimportant parameter while keeping the important one fixed. Random search explores the space of important parameters more thoroughly.

---

## 5. Further Reading
- [PyTorch Distributed Data Parallel](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)
- [A Visual Guide to Gradient Accumulation](https://towardsdatascience.com/what-is-gradient-accumulation-in-deep-learning-ec034122cfa)

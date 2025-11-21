# Day 10: PyTorch Lightning - Interview Questions

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Frameworks, Scalability, and Engineering

### 1. Why would you use PyTorch Lightning over raw PyTorch?
**Answer:**
*   **Standardization**: Enforces a structured organization (Model, Data, Trainer), making code readable and reproducible.
*   **Scalability**: Switching from CPU to Multi-GPU (DDP) or TPU requires changing one flag, not rewriting the training loop.
*   **Boilerplate**: Handles logging, checkpointing, and mixed precision automatically.

### 2. What is the difference between `LightningModule` and `nn.Module`?
**Answer:**
*   `nn.Module` defines the *architecture* (layers and forward pass).
*   `LightningModule` subclasses `nn.Module` but adds the *system* logic: `training_step`, `configure_optimizers`, `validation_step`. It is a self-contained system.

### 3. Explain how `training_step` differs from a standard PyTorch loop.
**Answer:**
*   In raw PyTorch, you write the `for` loop, zero_grad, backward, and step manually.
*   In `training_step`, you only calculate and return the loss. The `Trainer` handles the gradients, optimization, and logging behind the scenes.

### 4. How does Lightning handle Distributed Data Parallel (DDP)?
**Answer:**
*   It automatically wraps the model in `DistributedDataParallel`.
*   It handles process spawning (or launching).
*   It ensures data samplers are set to `DistributedSampler` so each GPU gets a different slice of data.
*   It syncs logging (only rank 0 logs).

### 5. What is a `LightningDataModule`?
**Answer:**
*   A class that encapsulates all data-related logic: Downloading, Splitting, Transforming, and creating DataLoaders.
*   Ensures that data processing is reproducible and decoupled from the model.

### 6. What is "Gradient Accumulation" and how do you use it in Lightning?
**Answer:**
*   Simulating a larger batch size by accumulating gradients over $N$ steps before updating weights.
*   In Lightning: `Trainer(accumulate_grad_batches=4)`. No code change needed in the model.

### 7. How do you implement "Early Stopping" in Lightning?
**Answer:**
*   Use a Callback.
*   `EarlyStopping(monitor='val_loss', patience=3)`.
*   Pass it to the `Trainer(callbacks=[...])`.

### 8. What is the difference between `setup()` and `prepare_data()` in a DataModule?
**Answer:**
*   `prepare_data()`: Run only on a single process (Rank 0). Used for downloading/tokenizing (writing to disk).
*   `setup()`: Run on *every* GPU. Used for loading data from disk, applying transforms, and splitting.

### 9. How does Lightning handle Mixed Precision (AMP)?
**Answer:**
*   `Trainer(precision="16-mixed")`.
*   It automatically wraps the forward pass in `autocast` and uses a `GradScaler` for the backward pass.

### 10. What is "Lightning Fabric"?
**Answer:**
*   A lower-level API than Lightning Trainer.
*   It gives you the "Device" and "Strategy" primitives but lets you write your own custom training loop.
*   Useful for researchers who need full control but want DDP to be easy.

### 11. How do you debug a Lightning model?
**Answer:**
*   `fast_dev_run=True`: Runs 1 batch of train/val/test to check for crashes.
*   `overfit_batches=1`: Overfits on a single batch to check convergence.
*   `profiler="simple"`: Checks for bottlenecks.

### 12. Can you use a standard PyTorch `DataLoader` with Lightning?
**Answer:**
*   Yes. You can pass `train_dataloader` directly to `trainer.fit()`.
*   However, using a `DataModule` is recommended for better organization.

### 13. How do you access the current epoch or global step in Lightning?
**Answer:**
*   `self.current_epoch`
*   `self.global_step`
*   Available inside the `LightningModule`.

### 14. What is the `on_train_epoch_end` hook?
**Answer:**
*   A method in `LightningModule` called at the end of every training epoch.
*   Useful for logging epoch-level metrics (like average accuracy) or resetting custom counters.

### 15. How does Lightning handle "Sanity Checking"?
**Answer:**
*   By default, it runs 2 steps of validation *before* training starts.
*   This catches bugs in the validation loop immediately, saving you from waiting for an entire training epoch to crash.

### 16. What is `torch_xla` and how does Lightning use it?
**Answer:**
*   The library connecting PyTorch to Google TPUs.
*   Lightning abstracts it away. Setting `accelerator="tpu"` automatically switches backend to XLA without code changes.

### 17. How do you save/load checkpoints in Lightning?
**Answer:**
*   **Auto-save**: `ModelCheckpoint` callback saves top-k models based on a metric.
*   **Manual**: `trainer.save_checkpoint("model.ckpt")`.
*   **Load**: `MyModel.load_from_checkpoint("model.ckpt")`.

### 18. What is the difference between `step` and `epoch` logging?
**Answer:**
*   `self.log("loss", loss, on_step=True, on_epoch=True)`.
*   `on_step`: Logs every iteration (noisy, good for debugging).
*   `on_epoch`: Aggregates (mean) over the epoch (smooth, good for trends).

### 19. How do you implement multiple optimizers (e.g., GANs) in Lightning?
**Answer:**
*   Return a list from `configure_optimizers`: `return [opt_g, opt_d], []`.
*   In `training_step`, use `optimizer_idx` argument to decide which logic to run (Generator vs Discriminator).

### 20. What is "Gradient Clipping" in Lightning?
**Answer:**
*   `Trainer(gradient_clip_val=1.0)`.
*   Automatically clips gradients before the optimizer step to prevent explosion.

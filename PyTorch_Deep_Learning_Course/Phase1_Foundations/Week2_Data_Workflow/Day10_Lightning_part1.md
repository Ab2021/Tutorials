# Day 10: PyTorch Lightning - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: DDP Strategies, Fabric, and TPU

## 1. Distributed Strategies (DDP vs FSDP)

Lightning handles the complexity of distributed training.
*   **DDP (Distributed Data Parallel)**: Copies model to each GPU. Syncs grads.
    *   `strategy="ddp"`
*   **DDP Spawn**: Spawns processes (slower, harder to debug).
*   **DeepSpeed / FSDP**: Shards model across GPUs. Needed for LLMs.
    *   `strategy="deepspeed_stage_3"`

## 2. Fabric: The Middle Ground

Some users find Lightning too opinionated.
**Lightning Fabric** gives you the mix-and-match power.
You write your own loop, but Fabric handles device placement.

```python
import lightning as L
fabric = L.Fabric(accelerator="cuda", devices=2)
fabric.launch()

model, optimizer = fabric.setup(model, optimizer)
dataloader = fabric.setup_dataloaders(dataloader)

model.train()
for batch in dataloader:
    loss = model(batch)
    fabric.backward(loss) # Handles scaling/mixed precision
    optimizer.step()
```

## 3. TPU Training

Training on Google Cloud TPUs (Tensor Processing Units) requires `torch_xla`.
Lightning abstracts this.
`trainer = L.Trainer(accelerator="tpu", devices=8)`
It handles the XLA compilation and data loading automatically.

## 4. Gradient Accumulation & Clipping

Built-in arguments.
```python
trainer = L.Trainer(
    accumulate_grad_batches=4,
    gradient_clip_val=1.0
)
```

## 5. Profiling in Lightning

Lightning has a built-in profiler.
```python
trainer = L.Trainer(profiler="simple")
```
It reports how much time is spent in `training_step`, `validation_step`, and `data_loading`.

# Day 5: The Training Loop - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 1 - The Engine
> **Topic**: AMP, GradScaler, and Performance Tuning

## 1. Automatic Mixed Precision (AMP)

Training in FP32 is stable but slow and memory-hungry.
Training in FP16 is fast but unstable (gradients underflow to 0).
**AMP** uses FP16 for heavy math (MatMul, Conv) and FP32 for sensitive ops (Loss, Softmax).

**Gradient Scaling**:
Since FP16 gradients are tiny, they might vanish (become 0).
We multiply the loss by a huge factor (e.g., 65536) before backward. Gradients become large (representable in FP16).
After backward, we divide gradients by the factor (unscale) before optimizer step.

```python
from torch.amp import autocast, GradScaler

scaler = GradScaler()

for inputs, targets in loader:
    optimizer.zero_grad()
    
    # Runs forward pass in Mixed Precision
    with autocast(device_type='cuda', dtype=torch.float16):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Scales loss. Calls backward() on scaled loss
    scaler.scale(loss).backward()
    
    # Unscales gradients and calls optimizer.step()
    scaler.step(optimizer)
    
    # Updates the scale for next iteration
    scaler.update()
```

## 2. Data Loading Bottlenecks

The GPU is a hungry beast. If the CPU cannot load/augment data fast enough, the GPU sits idle (0% utilization).
**Symptoms**: `nvidia-smi` shows volatile usage (100% -> 0% -> 100%).
**Fixes**:
*   `num_workers > 0` in DataLoader (Multi-processing).
*   `pin_memory=True` (Fast transfer).
*   Pre-fetch data.
*   Use `FFCV` or `WebDataset` for massive datasets.

## 3. Reproducibility

DL is non-deterministic (Random initialization, Shuffle, CUDA atomic ops).
To ensure same results:
```python
def seed_everything(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```
*Note*: Deterministic algorithms are slower.

## 4. Early Stopping

Stop training when Validation Loss stops improving. Prevents overfitting.

```python
best_loss = float('inf')
patience = 5
counter = 0

if val_loss < best_loss:
    best_loss = val_loss
    counter = 0
    torch.save(model.state_dict(), 'best.pt')
else:
    counter += 1
    if counter >= patience:
        print("Early Stopping!")
        break
```

## 5. Tqdm and Logging

Don't print every batch (spams console). Use `tqdm` for progress bars.
Use `TensorBoard` or `WandB` for plotting loss curves.

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

writer.add_scalar('Loss/train', loss, epoch)
```

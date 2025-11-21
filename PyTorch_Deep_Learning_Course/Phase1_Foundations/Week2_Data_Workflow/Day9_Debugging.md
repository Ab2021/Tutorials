# Day 9: Debugging & Visualization - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Hooks, TensorBoard, and Profiling

## 1. Theoretical Foundation: Observability

Deep Networks are "Black Boxes". To debug them, we need probes.
*   **Loss Curves**: The heartbeat of training.
*   **Histograms**: Distribution of weights/gradients. Detect saturation.
*   **Embeddings**: Visualizing high-dim space (t-SNE/PCA).
*   **Saliency Maps**: What is the model looking at?

## 2. PyTorch Hooks

Hooks are the surgical tools of PyTorch. They let you inject code into the Forward/Backward pass without changing the model definition.

### Forward Hooks
Signature: `hook(module, input, output)`
*   Use: Extract feature maps, check for NaNs in activations.

### Backward Hooks
Signature: `hook(module, grad_input, grad_output)`
*   Use: Check for Vanishing/Exploding gradients, Gradient Clipping.

```python
grads = {}
def save_grad(name):
    def hook(module, grad_input, grad_output):
        grads[name] = grad_output[0].detach()
    return hook

model.fc1.register_full_backward_hook(save_grad('fc1'))
loss.backward()
print(grads['fc1'].norm())
```

## 3. TensorBoard Visualization

PyTorch integrates with TensorBoard via `torch.utils.tensorboard`.

```python
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter('runs/experiment_1')

# 1. Scalars (Loss, Acc)
writer.add_scalar('Loss/train', loss, epoch)

# 2. Histograms (Weights)
for name, param in model.named_parameters():
    writer.add_histogram(name, param, epoch)

# 3. Images
writer.add_image('Input', img_grid, epoch)

# 4. Graph
writer.add_graph(model, dummy_input)

writer.close()
```
Run: `tensorboard --logdir=runs`

## 4. Common Bugs & Fixes

1.  **Silent Failures**: Model trains but accuracy is random.
    *   *Check*: Did you shuffle data? Is learning rate too high? Is loss function correct?
2.  **NaN Loss**:
    *   *Check*: Exploding gradients? Division by zero? `log(0)`?
    *   *Fix*: `torch.autograd.detect_anomaly()`.
3.  **Overfitting**:
    *   *Check*: Train loss low, Val loss high.
    *   *Fix*: Dropout, Weight Decay, More Data.

## 5. Profiling (Bottlenecks)

Is your code CPU bound or GPU bound?
`torch.profiler` tells you exactly where time is spent.

```python
with torch.profiler.profile(
    activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
    record_shapes=True
) as prof:
    model(inputs)

print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
```

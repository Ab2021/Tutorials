# Day 7: Transforms - Deep Dive

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: GPU Augmentation, Kornia, and Differentiable Transforms

## 1. CPU vs GPU Augmentation

Standard TorchVision transforms run on **CPU** (using PIL or NumPy).
**Bottleneck**:
1.  CPU decodes JPEG.
2.  CPU augments (Resize, Crop).
3.  CPU -> GPU Transfer.
4.  GPU trains.

If augmentation is heavy (large rotations, elastic deform), CPU becomes the bottleneck.

**GPU Augmentation**:
1.  CPU decodes JPEG -> GPU.
2.  GPU augments (Tensor operations).
3.  GPU trains.
*   Much faster because resizing/interpolation is parallelizable.

```python
# TorchVision v2 supports GPU transforms
transforms = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(...)
])
# Move data to GPU FIRST, then transform
img = img.cuda()
out = transforms(img)
```

## 2. Kornia: Differentiable Computer Vision

**Kornia** is a library for CV in PyTorch.
*   All transforms are `nn.Module` and differentiable.
*   You can backpropagate through the augmentation!
*   Useful for: Spatial Transformer Networks, Adversarial Attacks, Geometric Deep Learning.

```python
import kornia.augmentation as K

aug = K.AugmentationSequential(
    K.RandomAffine(degrees=360, p=1.0),
    K.ColorJitter(0.1, 0.1, 0.1, 0.1),
    data_keys=["input", "mask"]
)

img_batch = torch.randn(32, 3, 224, 224).cuda()
out = aug(img_batch) # Runs on GPU
```

## 3. MixUp and CutMix Implementation

These are "Batch" augmentations (operate on pairs of samples). Best implemented inside the training loop or collate function.

```python
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size).cuda()
    
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# In Loop:
inputs, targets_a, targets_b, lam = mixup_data(inputs, targets)
outputs = model(inputs)
loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
```

## 4. Interpolation Modes

Resizing requires interpolation.
*   **Nearest Neighbor**: Fast, blocky. Use for **Masks** (categorical).
*   **Bilinear**: Standard for images.
*   **Bicubic**: Smoother, slower.
*   **Antialiasing**: Crucial when downsampling to avoid artifacts. TorchVision v2 supports `antialias=True`.

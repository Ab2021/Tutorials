# Day 7: Transforms & Augmentation - Theory & Implementation

> **Phase**: 1 - Foundations
> **Week**: 2 - Data & Workflow
> **Topic**: Data Augmentation, TorchVision, and Invariance

## 1. Theoretical Foundation: Invariance vs Equivariance

Why do we augment data? To teach the model symmetries of the physical world.

### Invariance
The output should **not change** when input transforms.
*   **Classification**: A rotated cat is still a cat. $f(T(x)) = f(x)$.
*   We want the model to be invariant to translation, rotation, lighting, and noise.

### Equivariance
The output should **change in the same way** as the input.
*   **Segmentation**: If I rotate the image, the mask should rotate too. $f(T(x)) = T(f(x))$.
*   **Object Detection**: Bounding boxes must shift.

### The Regularization Effect
Augmentation artificially expands the dataset size.
It prevents the model from memorizing high-frequency noise or specific pixel arrangements.
It forces the model to learn robust, semantic features.

## 2. TorchVision Transforms v2

PyTorch recently released Transforms V2, which handles Segmentation Masks and Bounding Boxes automatically (Equivariance!).

```python
from torchvision.transforms import v2

transforms = v2.Compose([
    v2.ToImage(), # Convert to Tensor
    v2.RandomResizedCrop(size=(224, 224), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ColorJitter(brightness=0.2, contrast=0.2),
    v2.ToDtype(torch.float32, scale=True), # Normalize [0,1]
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Apply to Image AND Mask simultaneously
img, mask = transforms(img, mask)
```

## 3. Common Augmentations

### Geometric
*   **Crop**: Forces model to look at parts, not just global context.
*   **Flip**: Symmetry.
*   **Rotation/Affine**: Viewpoint invariance.

### Photometric (Color)
*   **Jitter**: Simulates different lighting/cameras.
*   **Grayscale**: Forces focus on shape/texture.

### Advanced (Mixing)
*   **MixUp**: Linear interpolation of two images and labels. $x' = \lambda x_1 + (1-\lambda) x_2$.
*   **CutMix**: Paste a patch of one image onto another.
*   **AutoAugment**: Learned policy of augmentations.

## 4. Test Time Augmentation (TTA)

During inference, we can average predictions across multiple augmented versions of the input.
*   Input: Image X.
*   Augments: X, Flip(X), Crop(X).
*   Output: Mean(Model(X), Model(Flip(X)), ...).
*   Boosts accuracy by 1-2% at cost of speed.

## 5. Normalization

Why subtract mean and divide by std?
*   **Centering**: Keeps gradients well-conditioned.
*   **Scaling**: Ensures features have similar ranges (loss surface is spherical, not elliptical).
*   **ImageNet Stats**: `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]` are standard.

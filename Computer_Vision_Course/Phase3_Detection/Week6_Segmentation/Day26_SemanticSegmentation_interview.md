# Day 26 Interview Questions: Semantic Segmentation

## Q1: What is the difference between Semantic and Instance Segmentation?
**Answer:**
*   **Semantic:** Classifies each pixel into a category (e.g., "Car"). All cars are labeled with the same color. Does not distinguish individual objects.
*   **Instance:** Detects and segments each individual object (e.g., "Car 1", "Car 2"). Distinguishes between instances of the same class.

## Q2: Explain Transposed Convolution. Is it the inverse of Convolution?
**Answer:**
*   It is **not** the mathematical inverse (it doesn't recover the original input values).
*   It recovers the **spatial shape** (upsampling).
*   It works by broadcasting input pixels to a larger area defined by the kernel, effectively reversing the downsampling operation of a stride > 1 convolution.

## Q3: Why does U-Net use concatenation instead of addition (like ResNet)?
**Answer:**
*   **Addition:** Fuses features by summing values. Good for residual learning (refinement).
*   **Concatenation:** Stacks features along the channel dimension.
    *   Preserves the high-res "where" information from the Encoder.
    *   Allows the Decoder to learn how to combine "what" (low-res semantic) and "where" (high-res spatial) features freely using its own convolutions.

## Q4: What are Checkerboard Artifacts?
**Answer:**
*   Grid-like patterns that appear in the output of Transposed Convolutions.
*   Caused when the kernel size is not divisible by the stride (overlap is uneven).
*   **Fix:** Use Resize (Nearest/Bilinear) + Convolution instead of Transposed Convolution.

## Q5: Why is Dice Loss better for medical imaging?
**Answer:**
*   Medical images often have extreme class imbalance (e.g., small lesion vs large background).
*   Dice Loss directly optimizes the **overlap** metric (F1 score), which is robust to imbalance.
*   Cross-Entropy can get stuck in a local minimum where it predicts everything as background.

## Q6: What is the receptive field requirement for segmentation?
**Answer:**
*   Ideally, the receptive field should cover the **entire image** (global context).
*   To classify a pixel as "sky", you need to know it's above the "ground".
*   FCN/U-Net achieve this via pooling layers (downsampling) which increase the receptive field rapidly.

## Q7: How do you handle input images of arbitrary sizes in FCN?
**Answer:**
*   FCN has no Fully Connected layers (which require fixed input size).
*   It only uses Convolutions and Pooling.
*   Therefore, it can accept any $H \times W$. The output will be a proportional map $H/32 \times W/32$ (before upsampling).

## Q8: What is mIoU?
**Answer:**
Mean Intersection over Union.
*   Compute IoU for each class: $TP / (TP + FP + FN)$.
*   Take the average over all classes.
*   The standard metric for semantic segmentation.

## Q9: Can U-Net be used for tasks other than segmentation?
**Answer:**
**Yes.**
*   U-Net is a general-purpose **Image-to-Image translation** architecture.
*   Used in: Denoising, Super-Resolution, Colorization, Depth Estimation, and Diffusion Models (Stable Diffusion uses a U-Net backbone).

## Q10: Implement Dice Loss in PyTorch.
**Answer:**
```python
def dice_loss(pred, target, smooth=1.):
    # pred: (B, H, W) probabilities
    # target: (B, H, W) binary mask
    
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(1,2))
    
    loss = 1 - ((2. * intersection + smooth) / 
                (pred.sum(dim=(1,2)) + target.sum(dim=(1,2)) + smooth))
    
    return loss.mean()
```

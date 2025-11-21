# Day 14: Object Detection - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: FPN, Focal Loss, and GIoU

## 1. Feature Pyramid Networks (FPN)

**Problem**: Detecting small objects.
*   Deep layers have high semantic value but low resolution (small objects disappear).
*   Shallow layers have high resolution but weak semantics.

**Solution**: Combine them.
1.  **Bottom-up**: Standard ResNet backbone.
2.  **Top-down**: Upsample deep features.
3.  **Lateral Connections**: Add upsampled deep features to shallow features ($1 \times 1$ conv).

Result: A pyramid of feature maps, all semantically strong. We predict boxes at *every* level.

## 2. Focal Loss (RetinaNet)

**Problem**: Class Imbalance.
In an image, 99% of anchors are Background (Easy Negatives). 1% are Objects.
Cross Entropy is overwhelmed by the easy negatives.

**Solution**: Down-weight easy examples.
$$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$
*   If $p_t \approx 1$ (Easy), $(1-p_t)^\gamma \approx 0$. Loss is zeroed out.
*   If $p_t$ is small (Hard), loss is preserved.
*   Allows One-Stage detectors to beat Two-Stage ones.

## 3. Generalized IoU (GIoU) Loss

MSE Loss for box coordinates $(x, y, w, h)$ is bad because it doesn't correlate with IoU.
Direct IoU Loss is bad because if boxes don't overlap, IoU is 0 and Gradient is 0.

**GIoU**:
$$ GIoU = IoU - \frac{Area(C) - Area(A \cup B)}{Area(C)} $$
*   $C$: Smallest enclosing box covering $A$ and $B$.
*   Provides gradients even when boxes are far apart (moves them closer).

## 4. Anchor-Free Detection (FCOS, CenterNet)

Anchors are annoying hyperparameters (Size? Aspect Ratio?).
**Anchor-Free**:
*   Treat detection as point estimation (Center of object).
*   Regress distance to 4 sides $(l, t, r, b)$ from the center.
*   Simpler, faster, and competitive accuracy.

## 5. DETR (Detection Transformer)

End-to-End detection with Transformers.
*   CNN Backbone $\to$ Transformer Encoder-Decoder.
*   **Bipartite Matching Loss**: Uses Hungarian Algorithm to match predictions to ground truth one-to-one.
*   No NMS needed! The attention mechanism learns to output unique boxes.

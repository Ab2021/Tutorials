# Day 22 Deep Dive: YOLO Evolution & SSD

## 1. YOLOv2 (2016) - "Better, Faster, Stronger"
*   **Batch Normalization:** Added to all conv layers.
*   **High-Res Classifier:** Fine-tuned on $448 \times 448$ ImageNet.
*   **Anchor Boxes:** Introduced anchors (priors) derived from K-Means clustering on the dataset.
*   **Passthrough Layer:** Concatenates high-res features with low-res features (early FPN).
*   **Multi-Scale Training:** Randomly resizing input images during training.

## 2. YOLOv3 (2018)
*   **Backbone:** Darknet-53 (Residual connections).
*   **3 Scales:** Predictions at 3 different scales (small, medium, large objects).
*   **Logistic Regression:** Used for objectness score instead of Softmax.

## 3. SSD (Single Shot MultiBox Detector)
**Idea:** Detect objects at multiple scales using feature maps from different layers.
*   **Pyramidal Feature Hierarchy:**
    *   Layer Conv4_3 ($38 \times 38$) $\to$ Detect small objects.
    *   Layer Conv7 ($19 \times 19$) $\to$ Detect medium objects.
    *   ...
    *   Layer Conv11 ($1 \times 1$) $\to$ Detect large objects.
*   **Priors (Anchors):** Pre-defined boxes at each location with different aspect ratios.

## 4. Focal Loss (RetinaNet)
**Problem:** One-stage detectors suffer from extreme class imbalance (100k background anchors vs 100 object anchors).
**Solution:** Down-weight easy examples (background).
$$ FL(p_t) = -(1 - p_t)^\gamma \log(p_t) $$
*   If $p_t \approx 1$ (easy), $(1-p_t)^\gamma \approx 0$. Loss is negligible.
*   Focuses training on hard misclassified examples.

## Summary
YOLO evolved by adopting best practices (Anchors, ResNet, FPN). SSD introduced multi-scale detection from different layers. RetinaNet solved the imbalance problem with Focal Loss.

# Day 23: Anchor-Free Detectors

## 1. The Problem with Anchors
Anchor-based detectors (YOLOv2+, RetinaNet) dominate but have issues:
1.  **Hyperparameters:** Sensitive to anchor size/ratio configuration.
2.  **Imbalance:** Generate massive number of negative anchors.
3.  **Complexity:** IoU calculation during training is slow.

## 2. FCOS (Fully Convolutional One-Stage)
**Idea:** Solve detection as per-pixel prediction (like segmentation).
*   **Target:** If a pixel $(x, y)$ falls inside a ground truth box, it is a positive sample.
*   **Regression:** Predict distance to 4 sides $(l, t, r, b)$ from the pixel.
*   **Centerness:** Predicts how close the pixel is to the center of the box. Used to down-weight low-quality predictions.

**Benefits:**
*   No anchor hyperparameters.
*   Simpler training pipeline.
*   Better performance than RetinaNet.

## 3. CenterNet: Objects as Points
**Idea:** Model an object as a single point (center of bounding box).
1.  **Heatmap:** Predict a heatmap $Y \in [0, 1]^{W/R \times H/R \times C}$ where peaks correspond to object centers.
2.  **Offset:** Predict local offset to recover discretization error caused by output stride.
3.  **Size:** Predict width and height $(w, h)$ at the center location.

**Inference:**
*   Find peaks in heatmap (max pooling).
*   Extract size and offset.
*   **No NMS needed!** (If peaks are sharp enough).

```python
# CenterNet Heatmap Loss (Modified Focal Loss)
def focal_loss(pred, gt):
    pos_inds = gt.eq(1)
    neg_inds = gt.lt(1)
    
    neg_weights = torch.pow(1 - gt, 4)
    
    loss = 0
    
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds
    
    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()
    
    if num_pos == 0:
        return -neg_loss
    return -(pos_loss + neg_loss) / num_pos
```

## 4. CornerNet
**Idea:** Detect Top-Left and Bottom-Right corners as keypoints.
*   **Embeddings:** Predict an embedding vector for each corner. Corners belonging to the same object should have similar embeddings.
*   **Corner Pooling:** Specialized pooling layer to find corners.

## Summary
Anchor-free methods simplified detection by removing the complex anchor design process, treating detection more like keypoint estimation or segmentation.

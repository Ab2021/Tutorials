# Day 14: Object Detection - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: YOLO, R-CNN, and Metrics

### 1. Explain the difference between One-Stage and Two-Stage detectors.
**Answer:**
*   **Two-Stage (Faster R-CNN)**: First generates Region Proposals (candidate boxes), then classifies/refines them. Slower, higher accuracy.
*   **One-Stage (YOLO)**: Dense prediction grid. Predicts class and box coordinates directly from feature maps in a single pass. Faster, slightly lower accuracy (historically).

### 2. What is "Intersection over Union" (IoU)?
**Answer:**
*   Area of Overlap / Area of Union.
*   Standard metric to evaluate how well a predicted box matches the ground truth.

### 3. How does Non-Maximum Suppression (NMS) work?
**Answer:**
*   Eliminates duplicate detections for the same object.
*   Iteratively selects the box with highest confidence and removes all other boxes that have high IoU (e.g., > 0.5) with it.

### 4. What are "Anchor Boxes"? Why do we need them?
**Answer:**
*   Pre-defined box shapes (priors) of different scales and aspect ratios.
*   The network predicts *offsets* from these anchors rather than absolute coordinates.
*   Simplifies learning (easier to adjust a template than draw from scratch).

### 5. What is "Focal Loss"?
**Answer:**
*   Designed to handle extreme class imbalance (Foreground vs Background).
*   Adds a modulating factor $(1-p)^\gamma$ to Cross Entropy.
*   Down-weights easy examples (background) so the model focuses on hard examples (objects).

### 6. Explain the YOLO output format.
**Answer:**
*   $S \times S$ grid.
*   Each cell predicts $B$ bounding boxes and $C$ class probabilities.
*   Tensor: $S \times S \times (B \times 5 + C)$.

### 7. What is "RoI Pooling" (or RoI Align)?
**Answer:**
*   Used in Two-Stage detectors.
*   Extracts a fixed-size feature map (e.g., $7 \times 7$) from a region of arbitrary size in the feature map.
*   **RoI Align** (Mask R-CNN) uses bilinear interpolation instead of quantization to be more precise (crucial for masks).

### 8. Why is MSE loss bad for Bounding Box regression?
**Answer:**
*   It is not scale-invariant. An error of 5 pixels is huge for a small box but negligible for a large box.
*   It doesn't correlate perfectly with IoU (the final metric).
*   GIoU or CIoU loss is preferred.

### 9. What is "Mean Average Precision" (mAP)?
**Answer:**
*   The area under the Precision-Recall curve, averaged over all classes and IoU thresholds (e.g., mAP@0.5:0.95).
*   The gold standard metric for detection.

### 10. How does FPN help with small objects?
**Answer:**
*   It fuses high-resolution features (early layers) with high-semantic features (late layers).
*   Predictions are made at multiple scales. Small objects are detected on the high-resolution levels of the pyramid.

### 11. What is the "Hungarian Matching Loss" in DETR?
**Answer:**
*   A set-based loss that forces a one-to-one mapping between predictions and ground truth.
*   Solves the duplicate prediction problem without NMS.

### 12. What is "CenterNet"?
**Answer:**
*   An anchor-free detector.
*   Models an object as a single point (center).
*   Uses a heatmap to predict centers and regresses size $(w, h)$.

### 13. What is the difference between YOLOv1, v3, and v5?
**Answer:**
*   **v1**: No anchors, simple grid.
*   **v3**: Added FPN (3 scales), Anchors, Logistic regression for classes.
*   **v5**: PyTorch implementation, Mosaic augmentation, Auto-learning anchors.

### 14. What is "Mosaic Augmentation"?
**Answer:**
*   Stitching 4 training images together into one large image.
*   Helps model learn context and detect small objects.
*   Allows Batch Normalization to see data from 4 different images at once.

### 15. How do you handle objects that span multiple grid cells in YOLO?
**Answer:**
*   The object is assigned to the grid cell that contains its **center**.
*   That specific cell is responsible for detecting it.

### 16. What is "Hard Negative Mining"?
**Answer:**
*   Explicitly selecting the hardest negative examples (backgrounds that look like objects) to train on.
*   Focal Loss makes this largely obsolete (automatic hard mining).

### 17. Why is Object Detection harder than Classification?
**Answer:**
*   Output is variable size (0 to N objects).
*   Localization requires regression (continuous), Classification is discrete.
*   Class imbalance (Background >> Objects).

### 18. What is "CIoU" (Complete IoU)?
**Answer:**
*   Improves GIoU by adding a term for **Aspect Ratio** consistency.
*   Considers: Overlap area, Center distance, and Aspect ratio.

### 19. How does RPN (Region Proposal Network) work?
**Answer:**
*   A small fully convolutional network that slides over the feature map.
*   At each location, it predicts "Objectness" score and box offsets for $k$ anchors.
*   Outputs proposals for the second stage.

### 20. What is the "Receptive Field" requirement for detection?
**Answer:**
*   The receptive field must be large enough to cover the object to recognize it.
*   Dilated convolutions or FPNs help increase effective RF.

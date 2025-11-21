# Day 22 Interview Questions: Single-Shot Detectors

## Q1: What is the main difference between Two-Stage (Faster R-CNN) and One-Stage (YOLO) detectors?
**Answer:**
*   **Two-Stage:**
    1.  Generate Region Proposals (RPN).
    2.  Classify and Refine proposals.
    *   Pros: Higher accuracy. Cons: Slower.
*   **One-Stage:**
    *   Predict bounding boxes and class probabilities directly from the feature map in a single pass.
    *   Pros: Real-time speed. Cons: Historically lower accuracy (struggled with small objects).

## Q2: How does YOLO handle multiple objects in the same grid cell?
**Answer:**
*   **YOLOv1:** Could only detect one object per cell (limitation).
*   **YOLOv2+:** Uses **Anchor Boxes**. Each grid cell predicts $B$ bounding boxes (e.g., 3 or 9) based on different anchor priors. It can detect multiple objects if they have different aspect ratios matching the anchors.

## Q3: Explain "Intersection over Union" (IoU).
**Answer:**
A metric to measure the overlap between two bounding boxes (Ground Truth $A$ and Prediction $B$).
$$ \text{IoU} = \frac{\text{Area}(A \cap B)}{\text{Area}(A \cup B)} $$
*   Used for assigning anchors to ground truth during training and for NMS during inference.

## Q4: What is Non-Maximum Suppression (NMS)?
**Answer:**
A post-processing step to remove duplicate detections.
1.  Sort predictions by confidence score.
2.  Pick the highest confidence box $M$.
3.  Discard any other box that has high IoU (e.g., > 0.5) with $M$.
4.  Repeat until no boxes remain.

## Q5: Why did YOLOv3 introduce predictions at 3 scales?
**Answer:**
To improve **small object detection**.
*   YOLOv1/v2 struggled with small objects because the final feature map was too coarse ($13 \times 13$).
*   YOLOv3 predicts boxes at $13 \times 13$ (large), $26 \times 26$ (medium), and $52 \times 52$ (small), similar to FPN.

## Q6: What is the "Class Imbalance" problem in one-stage detectors?
**Answer:**
*   The detector generates thousands of anchor boxes (e.g., 100k).
*   Only a few contain objects (Positives). The rest are background (Negatives).
*   The loss is dominated by the easy negatives, drowning out the gradient from the positives.
*   **Solution:** Focal Loss (RetinaNet) or Hard Negative Mining (SSD).

## Q7: How are Anchor Boxes determined in YOLO?
**Answer:**
By running **K-Means Clustering** on the width and height of the ground truth bounding boxes in the training set.
*   This ensures the priors match the statistics of the dataset (e.g., if most objects are tall pedestrians, anchors will be tall).

## Q8: What is the output tensor shape of YOLOv1?
**Answer:**
$S \times S \times (B \times 5 + C)$.
*   $S$: Grid size (e.g., 7).
*   $B$: Number of boxes per cell (e.g., 2).
*   $5$: $(x, y, w, h, \text{confidence})$.
*   $C$: Number of classes (e.g., 20).
*   Total: $7 \times 7 \times 30$.

## Q9: Why is "Centerness" used in FCOS?
**Answer:**
To down-weight low-quality bounding boxes predicted far from the center of an object.
*   Anchor-free detectors (FCOS) predict distance to 4 sides from a point. Points near the edge of an object often produce poor boxes.
*   Centerness branch predicts how close a pixel is to the center, used to filter predictions during NMS.

## Q10: Implement IoU calculation.
**Answer:**
```python
def iou(box1, box2):
    # box: (x1, y1, x2, y2)
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    
    box1_area = (box1[2]-box1[0]) * (box1[3]-box1[1])
    box2_area = (box2[2]-box2[0]) * (box2[3]-box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area
```

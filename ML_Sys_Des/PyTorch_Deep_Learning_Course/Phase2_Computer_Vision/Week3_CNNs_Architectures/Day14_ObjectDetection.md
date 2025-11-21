# Day 14: Object Detection - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: YOLO, R-CNN, IoU, and NMS

## 1. Theoretical Foundation: Classification vs Detection

*   **Classification**: "What is in the image?" (One label).
*   **Detection**: "What is where?" (Bounding Boxes + Labels).

### The Output Representation
For each object, we predict:
1.  **Class ($C$)**: Probability distribution.
2.  **Box ($B$)**: $(x, y, w, h)$. Center coordinates, width, height.

### Two Paradigms
1.  **Two-Stage (R-CNN family)**:
    *   Stage 1: Propose regions (RPN).
    *   Stage 2: Classify regions.
    *   *Pros*: Accurate. *Cons*: Slow.
2.  **One-Stage (YOLO, SSD)**:
    *   Predict boxes and classes directly from the feature map in one pass.
    *   *Pros*: Fast (Real-time). *Cons*: Historically less accurate (fixed by YOLOv4+).

## 2. Key Concepts

### Intersection over Union (IoU)
Metric to measure overlap between two boxes.
$$ IoU = \frac{Area(A \cap B)}{Area(A \cup B)} $$
*   IoU > 0.5 is usually a "Hit".

### Non-Maximum Suppression (NMS)
The model predicts thousands of boxes. Many overlap the same object.
**Algorithm**:
1.  Sort boxes by confidence.
2.  Pick the highest confidence box $B_{best}$.
3.  Discard all other boxes with $IoU(B_{best}, B_i) > Threshold$.
4.  Repeat.

### Anchor Boxes (Priors)
Instead of predicting width/height from scratch, we predict *offsets* from pre-defined box shapes (Anchors).
*   $w_{pred} = w_{anchor} \cdot e^{t_w}$
*   Helps the model specialize (Tall anchors for pedestrians, Wide for cars).

## 3. YOLO (You Only Look Once) Architecture

Divides image into $S \times S$ grid.
Each cell predicts $B$ boxes and $C$ class probabilities.
Output Tensor: $S \times S \times (B \times (5 + C))$.
*   5 comes from $(x, y, w, h, confidence)$.

## 4. Implementation: Simple Detection Loss

```python
import torch
import torch.nn as nn

def detection_loss(pred_box, true_box, pred_cls, true_cls):
    # 1. Localization Loss (MSE or GIoU)
    # Only penalize if object exists
    loc_loss = nn.MSELoss()(pred_box, true_box)
    
    # 2. Classification Loss (Cross Entropy)
    cls_loss = nn.CrossEntropyLoss()(pred_cls, true_cls)
    
    return loc_loss + cls_loss
```

## 5. Using TorchVision Faster R-CNN

```python
from torchvision.models.detection import fasterrcnn_resnet50_fpn

# Load pre-trained model
model = fasterrcnn_resnet50_fpn(weights="DEFAULT")
model.eval()

x = [torch.rand(3, 300, 400)] # List of images
predictions = model(x)

# Output: [{'boxes': ..., 'labels': ..., 'scores': ...}]
```

# Day 11: Object Detection Fundamentals

## Overview
Object detection extends classification by localizing objects with bounding boxes. This lesson covers the foundational concepts: sliding windows, region proposals, IoU, NMS, and evaluation metrics.

## 1. Problem Formulation

### Classification vs Detection vs Segmentation

**Classification:** What is in the image?
- Input: Image
- Output: Class label
- Example: "Cat"

**Object Detection:** What and where?
- Input: Image
- Output: Bounding boxes + class labels
- Example: [(x1, y1, x2, y2, "Cat"), (x3, y3, x4, y4, "Dog")]

**Segmentation:** Pixel-level classification
- Input: Image
- Output: Pixel-wise labels
- Example: Mask for each object

### Bounding Box Representation

**Formats:**
1. **Corner format:** $(x_{min}, y_{min}, x_{max}, y_{max})$
2. **Center format:** $(x_{center}, y_{center}, width, height)$
3. **YOLO format:** $(x_{center}, y_{center}, w, h)$ normalized to [0,1]

```python
import numpy as np

def convert_bbox_format(bbox, from_format='xyxy', to_format='xywh'):
    """Convert between bounding box formats."""
    if from_format == 'xyxy' and to_format == 'xywh':
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        x_center = x1 + w / 2
        y_center = y1 + h / 2
        return [x_center, y_center, w, h]
    
    elif from_format == 'xywh' and to_format == 'xyxy':
        x_center, y_center, w, h = bbox
        x1 = x_center - w / 2
        y1 = y_center - h / 2
        x2 = x_center + w / 2
        y2 = y_center + h / 2
        return [x1, y1, x2, y2]
    
    else:
        raise ValueError(f"Unsupported conversion: {from_format} to {to_format}")

# Example
bbox_xyxy = [100, 150, 200, 300]
bbox_xywh = convert_bbox_format(bbox_xyxy, 'xyxy', 'xywh')
print(f"XYXY: {bbox_xyxy}")
print(f"XYWH: {bbox_xywh}")
```

## 2. Intersection over Union (IoU)

**Definition:** Overlap between predicted and ground truth boxes.

$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_p \cap B_{gt}|}{|B_p \cup B_{gt}|} $$

**Properties:**
- Range: [0, 1]
- 0: No overlap
- 1: Perfect overlap

### Implementation

```python
def compute_iou(box1, box2):
    """
    Compute IoU between two boxes.
    
    Args:
        box1, box2: [x1, y1, x2, y2] format
    
    Returns:
        iou: Intersection over Union
    """
    # Intersection coordinates
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])
    
    # Intersection area
    inter_width = max(0, x2_inter - x1_inter)
    inter_height = max(0, y2_inter - y1_inter)
    inter_area = inter_width * inter_height
    
    # Union area
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou

def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU matrix between two sets of boxes.
    
    Args:
        boxes1: (N, 4) array
        boxes2: (M, 4) array
    
    Returns:
        iou_matrix: (N, M) array
    """
    N = len(boxes1)
    M = len(boxes2)
    iou_matrix = np.zeros((N, M))
    
    for i in range(N):
        for j in range(M):
            iou_matrix[i, j] = compute_iou(boxes1[i], boxes2[j])
    
    return iou_matrix

# Vectorized version (faster)
def compute_iou_vectorized(boxes1, boxes2):
    """Vectorized IoU computation."""
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    
    # Expand dimensions for broadcasting
    boxes1 = boxes1[:, None, :]  # (N, 1, 4)
    boxes2 = boxes2[None, :, :]  # (1, M, 4)
    
    # Intersection
    x1_inter = np.maximum(boxes1[..., 0], boxes2[..., 0])
    y1_inter = np.maximum(boxes1[..., 1], boxes2[..., 1])
    x2_inter = np.minimum(boxes1[..., 2], boxes2[..., 2])
    y2_inter = np.minimum(boxes1[..., 3], boxes2[..., 3])
    
    inter_area = np.maximum(0, x2_inter - x1_inter) * np.maximum(0, y2_inter - y1_inter)
    
    # Areas
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])
    
    # Union
    union_area = boxes1_area + boxes2_area - inter_area
    
    # IoU
    iou = inter_area / (union_area + 1e-6)
    
    return iou.squeeze()
```

### IoU Variants

**Generalized IoU (GIoU):**
$$ \text{GIoU} = \text{IoU} - \frac{|C - (B_p \cup B_{gt})|}{|C|} $$

where $C$ is the smallest enclosing box.

**Distance IoU (DIoU):**
$$ \text{DIoU} = \text{IoU} - \frac{\rho^2(b_p, b_{gt})}{c^2} $$

where $\rho$ is the Euclidean distance between box centers, $c$ is the diagonal of the smallest enclosing box.

**Complete IoU (CIoU):**
$$ \text{CIoU} = \text{DIoU} - \alpha v $$

where $v$ measures aspect ratio consistency.

## 3. Non-Maximum Suppression (NMS)

**Problem:** Multiple overlapping detections for the same object.
**Solution:** Keep only the highest confidence detection, suppress others.

**Algorithm:**
1. Sort detections by confidence (descending)
2. Select detection with highest confidence
3. Remove all detections with IoU > threshold
4. Repeat until no detections remain

```python
def non_max_suppression(boxes, scores, iou_threshold=0.5):
    """
    Non-Maximum Suppression.
    
    Args:
        boxes: (N, 4) array [x1, y1, x2, y2]
        scores: (N,) array of confidence scores
        iou_threshold: IoU threshold for suppression
    
    Returns:
        keep_indices: Indices of boxes to keep
    """
    # Sort by score
    sorted_indices = np.argsort(scores)[::-1]
    
    keep_indices = []
    
    while len(sorted_indices) > 0:
        # Keep highest scoring box
        current = sorted_indices[0]
        keep_indices.append(current)
        
        if len(sorted_indices) == 1:
            break
        
        # Compute IoU with remaining boxes
        current_box = boxes[current]
        remaining_boxes = boxes[sorted_indices[1:]]
        
        ious = np.array([compute_iou(current_box, box) for box in remaining_boxes])
        
        # Keep boxes with IoU < threshold
        keep_mask = ious < iou_threshold
        sorted_indices = sorted_indices[1:][keep_mask]
    
    return keep_indices

# Vectorized NMS (faster)
def nms_vectorized(boxes, scores, iou_threshold=0.5):
    """Vectorized NMS implementation."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    
    keep = []
    
    while order.size > 0:
        i = order[0]
        keep.append(i)
        
        # Compute IoU with remaining boxes
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        
        # Keep boxes with IoU < threshold
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    
    return keep
```

### Soft-NMS

**Problem:** Hard NMS may remove correct detections of nearby objects.
**Solution:** Decay scores instead of removing.

$$ s_i = \begin{cases}
s_i & \text{IoU}(M, b_i) < N_t \\
s_i (1 - \text{IoU}(M, b_i)) & \text{IoU}(M, b_i) \geq N_t
\end{cases} $$

```python
def soft_nms(boxes, scores, iou_threshold=0.5, sigma=0.5, score_threshold=0.001):
    """Soft-NMS implementation."""
    N = len(boxes)
    
    for i in range(N):
        # Find box with max score
        max_idx = i + np.argmax(scores[i:])
        
        # Swap
        boxes[[i, max_idx]] = boxes[[max_idx, i]]
        scores[[i, max_idx]] = scores[[max_idx, i]]
        
        # Decay scores of overlapping boxes
        for j in range(i + 1, N):
            iou = compute_iou(boxes[i], boxes[j])
            
            # Gaussian decay
            scores[j] *= np.exp(-(iou ** 2) / sigma)
            
            # Linear decay (alternative)
            # if iou > iou_threshold:
            #     scores[j] *= (1 - iou)
    
    # Filter by score threshold
    keep = scores > score_threshold
    
    return boxes[keep], scores[keep]
```

## 4. Evaluation Metrics

### Precision and Recall

**True Positive (TP):** Correct detection (IoU > threshold)
**False Positive (FP):** Incorrect detection
**False Negative (FN):** Missed ground truth

$$ \text{Precision} = \frac{TP}{TP + FP} $$
$$ \text{Recall} = \frac{TP}{TP + FN} $$

### Average Precision (AP)

**AP:** Area under Precision-Recall curve.

```python
def compute_ap(precisions, recalls):
    """Compute Average Precision."""
    # Add sentinel values
    precisions = np.concatenate(([0], precisions, [0]))
    recalls = np.concatenate(([0], recalls, [1]))
    
    # Compute envelope
    for i in range(len(precisions) - 1, 0, -1):
        precisions[i - 1] = max(precisions[i - 1], precisions[i])
    
    # Find points where recall changes
    indices = np.where(recalls[1:] != recalls[:-1])[0]
    
    # Compute AP
    ap = np.sum((recalls[indices + 1] - recalls[indices]) * precisions[indices + 1])
    
    return ap

def evaluate_detection(pred_boxes, pred_scores, pred_labels,
                      gt_boxes, gt_labels, iou_threshold=0.5):
    """Evaluate object detection."""
    # Sort predictions by score
    sorted_indices = np.argsort(pred_scores)[::-1]
    pred_boxes = pred_boxes[sorted_indices]
    pred_labels = pred_labels[sorted_indices]
    pred_scores = pred_scores[sorted_indices]
    
    # Match predictions to ground truth
    tp = np.zeros(len(pred_boxes))
    fp = np.zeros(len(pred_boxes))
    
    gt_matched = np.zeros(len(gt_boxes), dtype=bool)
    
    for i, (pred_box, pred_label) in enumerate(zip(pred_boxes, pred_labels)):
        # Find ground truth boxes of same class
        same_class_mask = gt_labels == pred_label
        
        if not np.any(same_class_mask):
            fp[i] = 1
            continue
        
        # Compute IoU with ground truth
        ious = np.array([compute_iou(pred_box, gt_box) 
                        for gt_box in gt_boxes[same_class_mask]])
        
        # Find best match
        max_iou_idx = np.argmax(ious)
        max_iou = ious[max_iou_idx]
        
        # Map back to original index
        same_class_indices = np.where(same_class_mask)[0]
        gt_idx = same_class_indices[max_iou_idx]
        
        if max_iou >= iou_threshold and not gt_matched[gt_idx]:
            tp[i] = 1
            gt_matched[gt_idx] = True
        else:
            fp[i] = 1
    
    # Compute precision and recall
    tp_cumsum = np.cumsum(tp)
    fp_cumsum = np.cumsum(fp)
    
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum)
    recalls = tp_cumsum / len(gt_boxes)
    
    # Compute AP
    ap = compute_ap(precisions, recalls)
    
    return ap, precisions, recalls
```

### Mean Average Precision (mAP)

**mAP:** Average of AP across all classes.

$$ \text{mAP} = \frac{1}{C} \sum_{c=1}^C \text{AP}_c $$

**mAP@0.5:** IoU threshold = 0.5
**mAP@[0.5:0.95]:** Average over IoU thresholds from 0.5 to 0.95 (COCO metric)

## 5. Sliding Window Detection

**Naive approach:** Apply classifier at every location and scale.

```python
def sliding_window_detection(image, classifier, window_size=(64, 64), stride=8):
    """Sliding window object detection."""
    H, W = image.shape[:2]
    h, w = window_size
    
    detections = []
    
    for y in range(0, H - h + 1, stride):
        for x in range(0, W - w + 1, stride):
            # Extract window
            window = image[y:y+h, x:x+w]
            
            # Classify
            score, label = classifier(window)
            
            if score > 0.5:  # Threshold
                detections.append({
                    'box': [x, y, x+w, y+h],
                    'score': score,
                    'label': label
                })
    
    return detections
```

**Problems:**
- Computationally expensive
- Fixed aspect ratio
- Many redundant computations

**Solution:** Region proposals (Selective Search, RPN).

## Summary
Object detection requires localizing objects with bounding boxes. Key concepts include IoU for overlap measurement, NMS for removing duplicates, and mAP for evaluation.

**Next:** R-CNN family (R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN).

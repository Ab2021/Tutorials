# Day 11 Interview Questions: Object Detection Fundamentals

## Q1: Explain IoU and why it's important for object detection.
**Answer:**

**IoU (Intersection over Union):**
$$ \text{IoU} = \frac{\text{Area of Overlap}}{\text{Area of Union}} = \frac{|B_1 \cap B_2|}{|B_1 \cup B_2|} $$

**Importance:**
1. **Matching:** Determine if prediction matches ground truth
2. **Evaluation:** Compute precision/recall (TP if IoU > threshold)
3. **NMS:** Remove duplicate detections
4. **Training:** Assign anchors to ground truth

**Thresholds:**
- IoU > 0.5: Typically considered a match
- COCO: Uses IoU from 0.5 to 0.95 (stricter)

**Example:**
```python
pred_box = [100, 100, 200, 200]  # 100x100 area = 10,000
gt_box = [150, 150, 250, 250]    # 100x100 area = 10,000

# Intersection: [150, 150, 200, 200] = 50x50 = 2,500
# Union: 10,000 + 10,000 - 2,500 = 17,500
# IoU = 2,500 / 17,500 = 0.143
```

**Limitations:**
- IoU = 0 for non-overlapping boxes (no gradient)
- Doesn't capture distance between boxes
- Solutions: GIoU, DIoU, CIoU

## Q2: How does Non-Maximum Suppression work?
**Answer:**

**Problem:** Multiple detections for same object.

**NMS Algorithm:**
1. Sort all detections by confidence score (descending)
2. Select detection with highest score
3. Remove all detections with IoU > threshold with selected detection
4. Repeat until no detections remain

**Pseudocode:**
```
NMS(boxes, scores, iou_threshold):
    keep = []
    sorted_indices = argsort(scores, descending=True)
    
    while sorted_indices not empty:
        current = sorted_indices[0]
        keep.append(current)
        
        # Compute IoU with remaining boxes
        ious = compute_iou(boxes[current], boxes[sorted_indices[1:]])
        
        # Keep boxes with IoU < threshold
        sorted_indices = sorted_indices[1:][ious < iou_threshold]
    
    return keep
```

**Parameters:**
- **IoU threshold:** Typically 0.5
  - Too low: Remove valid detections
  - Too high: Keep duplicates

**Variants:**
- **Soft-NMS:** Decay scores instead of removing
- **Class-specific NMS:** Apply per class
- **NMS-free:** Learn to suppress duplicates (DETR)

## Q3: Compare one-stage vs two-stage detectors.
**Answer:**

| Aspect | Two-Stage (R-CNN) | One-Stage (YOLO, SSD) |
|--------|-------------------|----------------------|
| **Pipeline** | 1. Proposals 2. Classification | Direct prediction |
| **Speed** | Slower (10-30 FPS) | Faster (30-150 FPS) |
| **Accuracy** | Higher (mAP ~40-50%) | Lower (mAP ~30-45%) |
| **Small objects** | Better | Worse |
| **Class imbalance** | Balanced (RoI pooling) | Imbalanced (focal loss helps) |
| **Complexity** | More complex | Simpler |
| **Use case** | Accuracy priority | Speed priority |

**Two-Stage Example (Faster R-CNN):**
```
Image → CNN → RPN (proposals) → RoI Pooling → Classification + Regression
```

**One-Stage Example (YOLO):**
```
Image → CNN → Direct prediction of boxes + classes
```

**Modern trend:** One-stage detectors closing accuracy gap (RetinaNet, EfficientDet).

## Q4: Explain Average Precision (AP) and mAP.
**Answer:**

**Precision-Recall Curve:**
- **Precision:** $\frac{TP}{TP + FP}$ (how many detections are correct)
- **Recall:** $\frac{TP}{TP + FN}$ (how many ground truths are found)

**Average Precision (AP):**
Area under precision-recall curve.

**Computation:**
1. Sort predictions by confidence
2. For each prediction:
   - If IoU > threshold with unmatched GT: TP
   - Else: FP
3. Compute precision and recall at each point
4. Integrate area under curve

**11-point interpolation (PASCAL VOC):**
$$ AP = \frac{1}{11} \sum_{r \in \{0, 0.1, ..., 1.0\}} \max_{r' \geq r} p(r') $$

**All-point interpolation (COCO):**
$$ AP = \sum_{n} (r_n - r_{n-1}) p_{interp}(r_n) $$

**Mean Average Precision (mAP):**
$$ mAP = \frac{1}{C} \sum_{c=1}^C AP_c $$

**COCO metrics:**
- **mAP@0.5:** IoU threshold = 0.5
- **mAP@0.75:** IoU threshold = 0.75
- **mAP@[0.5:0.95]:** Average over IoU 0.5 to 0.95 (primary metric)

**Example:**
```
Class: Car
Predictions (sorted by confidence):
[0.9, TP], [0.8, TP], [0.7, FP], [0.6, TP], [0.5, FP]

Precision: [1.0, 1.0, 0.67, 0.75, 0.6]
Recall:    [0.33, 0.67, 0.67, 1.0, 1.0]

AP ≈ 0.85 (area under curve)
```

## Q5: What are anchor boxes and why use them?
**Answer:**

**Anchor Boxes:** Pre-defined boxes at different scales and aspect ratios.

**Why use anchors:**
1. **Handle multiple scales:** Small, medium, large objects
2. **Handle aspect ratios:** Square, wide, tall objects
3. **Reference for regression:** Easier to learn offsets than absolute coordinates
4. **Dense predictions:** One prediction per spatial location

**Generation:**
```python
# Base size: 16
# Scales: [8, 16, 32] → sizes [128, 256, 512]
# Ratios: [0.5, 1, 2] → aspect ratios

# Total: 3 scales × 3 ratios = 9 anchors per location
```

**Matching to ground truth:**
- **Positive:** IoU > 0.7 with any GT
- **Negative:** IoU < 0.3 with all GT
- **Ignore:** 0.3 ≤ IoU ≤ 0.7

**Regression targets:**
$$ t_x = \frac{x_{gt} - x_a}{w_a}, \quad t_w = \log\frac{w_{gt}}{w_a} $$

**Advantages:**
- Handles multiple objects per location
- Efficient (no region proposals needed)
- End-to-end trainable

**Disadvantages:**
- Hyperparameter tuning (scales, ratios)
- Class imbalance (many negative anchors)
- Fixed aspect ratios may not fit all objects

## Q6: Implement bounding box encoding and decoding.
**Answer:**

```python
import numpy as np

def encode_boxes(gt_boxes, anchors):
    """
    Encode ground truth boxes relative to anchors.
    
    Args:
        gt_boxes: (N, 4) [x1, y1, x2, y2]
        anchors: (N, 4) [x1, y1, x2, y2]
    
    Returns:
        targets: (N, 4) [tx, ty, tw, th]
    """
    # Convert to center format
    gt_w = gt_boxes[:, 2] - gt_boxes[:, 0]
    gt_h = gt_boxes[:, 3] - gt_boxes[:, 1]
    gt_cx = gt_boxes[:, 0] + 0.5 * gt_w
    gt_cy = gt_boxes[:, 1] + 0.5 * gt_h
    
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    
    # Encode
    tx = (gt_cx - anchor_cx) / anchor_w
    ty = (gt_cy - anchor_cy) / anchor_h
    tw = np.log(gt_w / anchor_w)
    th = np.log(gt_h / anchor_h)
    
    targets = np.stack([tx, ty, tw, th], axis=1)
    
    return targets

def decode_boxes(deltas, anchors):
    """
    Decode predicted deltas to boxes.
    
    Args:
        deltas: (N, 4) [tx, ty, tw, th]
        anchors: (N, 4) [x1, y1, x2, y2]
    
    Returns:
        boxes: (N, 4) [x1, y1, x2, y2]
    """
    # Anchor properties
    anchor_w = anchors[:, 2] - anchors[:, 0]
    anchor_h = anchors[:, 3] - anchors[:, 1]
    anchor_cx = anchors[:, 0] + 0.5 * anchor_w
    anchor_cy = anchors[:, 1] + 0.5 * anchor_h
    
    # Decode
    tx, ty, tw, th = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    
    pred_cx = tx * anchor_w + anchor_cx
    pred_cy = ty * anchor_h + anchor_cy
    pred_w = np.exp(tw) * anchor_w
    pred_h = np.exp(th) * anchor_h
    
    # Convert to corner format
    pred_boxes = np.zeros_like(deltas)
    pred_boxes[:, 0] = pred_cx - 0.5 * pred_w  # x1
    pred_boxes[:, 1] = pred_cy - 0.5 * pred_h  # y1
    pred_boxes[:, 2] = pred_cx + 0.5 * pred_w  # x2
    pred_boxes[:, 3] = pred_cy + 0.5 * pred_h  # y2
    
    return pred_boxes

# Example
gt_box = np.array([[100, 100, 200, 200]])
anchor = np.array([[90, 90, 210, 210]])

# Encode
targets = encode_boxes(gt_box, anchor)
print(f"Encoded: {targets}")  # Small offsets

# Decode
decoded = decode_boxes(targets, anchor)
print(f"Decoded: {decoded}")  # Should match gt_box
print(f"Match: {np.allclose(decoded, gt_box)}")
```

## Q7: What is Focal Loss and why is it needed?
**Answer:**

**Problem:** Class imbalance in one-stage detectors.
- Thousands of anchors per image
- Most are background (negative)
- Easy negatives dominate loss

**Cross-Entropy Loss:**
$$ CE(p_t) = -\log(p_t) $$

**Focal Loss:**
$$ FL(p_t) = -(1-p_t)^\gamma \log(p_t) $$

where $\gamma$ is focusing parameter (typically 2).

**Effect:**
- **Easy examples** ($p_t$ high): $(1-p_t)^\gamma$ small → down-weighted
- **Hard examples** ($p_t$ low): $(1-p_t)^\gamma$ large → up-weighted

**Example:**
```
p_t = 0.9 (easy):  (1-0.9)^2 = 0.01 → loss reduced 100×
p_t = 0.5 (hard):  (1-0.5)^2 = 0.25 → loss reduced 4×
p_t = 0.1 (very hard): (1-0.1)^2 = 0.81 → loss almost unchanged
```

**With class balancing:**
$$ FL(p_t) = -\alpha_t (1-p_t)^\gamma \log(p_t) $$

where $\alpha_t$ is class weight (typically 0.25 for foreground).

**Results:**
- RetinaNet with Focal Loss: mAP 39.1% (COCO)
- Same architecture with CE: mAP 31.2%
- **Improvement:** +7.9% mAP

## Q8: Explain Feature Pyramid Networks (FPN).
**Answer:**

**Motivation:** Detect objects at multiple scales efficiently.

**Problem with single-scale features:**
- Small objects: Need high-resolution features
- Large objects: Need semantic features
- Can't have both in single layer

**FPN Architecture:**

**1. Bottom-up pathway:**
Standard CNN (e.g., ResNet) producing features at multiple scales:
- C2: stride 4, 256 channels
- C3: stride 8, 512 channels
- C4: stride 16, 1024 channels
- C5: stride 32, 2048 channels

**2. Top-down pathway:**
Upsample coarse features and merge with fine features.

**3. Lateral connections:**
1×1 convolutions to match channels, then element-wise addition.

**Formula:**
$$ P_i = \text{Conv}_{1 \times 1}(C_i) + \text{Upsample}(P_{i+1}) $$

**Benefits:**
- **Multi-scale:** Features at multiple resolutions
- **Semantic:** All levels have high-level semantics
- **Efficient:** Shared computation
- **Performance:** +2-3% mAP improvement

**Usage:**
- P2: Detect small objects (stride 4)
- P3: Detect medium objects (stride 8)
- P4: Detect large objects (stride 16)
- P5: Detect very large objects (stride 32)

## Q9: How to handle class imbalance in object detection?
**Answer:**

**Problem:** Imbalance between foreground and background.
- Typical ratio: 1:1000 (positive:negative)
- Easy negatives dominate training

**Solutions:**

**1. Hard Negative Mining:**
Select hard negatives based on loss.
```python
# Keep top-k hard negatives
neg_loss, _ = loss[neg_mask].sort(descending=True)
num_neg = min(3 * num_pos, len(neg_loss))
hard_neg_loss = neg_loss[:num_neg]
```

**2. Focal Loss:**
Down-weight easy examples.
```python
focal_loss = -(1 - p_t) ** gamma * log(p_t)
```

**3. OHEM (Online Hard Example Mining):**
Forward pass → select hard examples → backward pass.

**4. Balanced Sampling:**
Sample equal positives and negatives per batch.

**5. Class Weights:**
Weight loss by inverse class frequency.
```python
weight = 1.0 / class_frequency
loss = weight * cross_entropy_loss
```

**6. Anchor Matching Strategy:**
- Positive: IoU > 0.7
- Negative: IoU < 0.3
- Ignore: 0.3 ≤ IoU ≤ 0.7 (don't train on ambiguous)

**Best practice:** Combine multiple strategies (e.g., Focal Loss + Hard Negative Mining).

## Q10: Design an object detection evaluation pipeline.
**Answer:**

```python
class DetectionEvaluator:
    """Evaluate object detection performance."""
    
    def __init__(self, num_classes, iou_thresholds=[0.5, 0.75, 0.95]):
        self.num_classes = num_classes
        self.iou_thresholds = iou_thresholds
        self.predictions = []
        self.ground_truths = []
    
    def add_predictions(self, boxes, scores, labels, image_id):
        """Add predictions for one image."""
        for box, score, label in zip(boxes, scores, labels):
            self.predictions.append({
                'image_id': image_id,
                'box': box,
                'score': score,
                'label': label
            })
    
    def add_ground_truths(self, boxes, labels, image_id):
        """Add ground truths for one image."""
        for box, label in zip(boxes, labels):
            self.ground_truths.append({
                'image_id': image_id,
                'box': box,
                'label': label,
                'matched': False
            })
    
    def compute_ap(self, class_id, iou_threshold):
        """Compute AP for one class at one IoU threshold."""
        # Filter predictions and GTs for this class
        class_preds = [p for p in self.predictions if p['label'] == class_id]
        class_gts = [g for g in self.ground_truths if g['label'] == class_id]
        
        if len(class_gts) == 0:
            return 0.0
        
        # Sort predictions by score
        class_preds.sort(key=lambda x: x['score'], reverse=True)
        
        # Match predictions to GTs
        tp = np.zeros(len(class_preds))
        fp = np.zeros(len(class_preds))
        
        gt_matched = {g['image_id']: [False] * len([x for x in class_gts if x['image_id'] == g['image_id']]) 
                     for g in class_gts}
        
        for i, pred in enumerate(class_preds):
            # Find GTs in same image
            image_gts = [g for g in class_gts if g['image_id'] == pred['image_id']]
            
            if len(image_gts) == 0:
                fp[i] = 1
                continue
            
            # Compute IoU with all GTs
            ious = [compute_iou(pred['box'], gt['box']) for gt in image_gts]
            max_iou_idx = np.argmax(ious)
            max_iou = ious[max_iou_idx]
            
            # Check if match
            if max_iou >= iou_threshold and not image_gts[max_iou_idx]['matched']:
                tp[i] = 1
                image_gts[max_iou_idx]['matched'] = True
            else:
                fp[i] = 1
        
        # Compute precision and recall
        tp_cumsum = np.cumsum(tp)
        fp_cumsum = np.cumsum(fp)
        
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum + 1e-6)
        recalls = tp_cumsum / len(class_gts)
        
        # Compute AP (11-point interpolation)
        ap = 0.0
        for t in np.linspace(0, 1, 11):
            if np.sum(recalls >= t) == 0:
                p = 0
            else:
                p = np.max(precisions[recalls >= t])
            ap += p / 11
        
        return ap
    
    def evaluate(self):
        """Compute mAP across all classes and IoU thresholds."""
        results = {}
        
        for iou_thresh in self.iou_thresholds:
            aps = []
            for class_id in range(self.num_classes):
                ap = self.compute_ap(class_id, iou_thresh)
                aps.append(ap)
            
            results[f'mAP@{iou_thresh}'] = np.mean(aps)
        
        # COCO-style mAP (average over IoU thresholds)
        results['mAP@[0.5:0.95]'] = np.mean([results[f'mAP@{t}'] 
                                             for t in self.iou_thresholds])
        
        return results

# Usage
evaluator = DetectionEvaluator(num_classes=80)

for image_id, (pred_boxes, pred_scores, pred_labels, gt_boxes, gt_labels) in enumerate(dataset):
    evaluator.add_predictions(pred_boxes, pred_scores, pred_labels, image_id)
    evaluator.add_ground_truths(gt_boxes, gt_labels, image_id)

results = evaluator.evaluate()
print(f"mAP@0.5: {results['mAP@0.5']:.3f}")
print(f"mAP@0.75: {results['mAP@0.75']:.3f}")
print(f"mAP@[0.5:0.95]: {results['mAP@[0.5:0.95]']:.3f}")
```

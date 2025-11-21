# Day 29 Interview Questions: Panoptic Segmentation

## Q1: What is the difference between "Things" and "Stuff"?
**Answer:**
*   **Things:** Countable objects with defined geometry (Car, Person, Chair). Handled by Instance Segmentation.
*   **Stuff:** Amorphous regions with no defined shape or instance count (Sky, Road, Grass, Water). Handled by Semantic Segmentation.

## Q2: Why can't Mask R-CNN do Panoptic Segmentation out of the box?
**Answer:**
*   Mask R-CNN is designed for **Instance Segmentation**.
*   It detects objects (Things) and masks them.
*   It ignores the background (Stuff). It doesn't output a label for the "Road" or "Sky".
*   Panoptic FPN adds a separate branch to handle this.

## Q3: Explain the Panoptic Quality (PQ) metric.
**Answer:**
*   $PQ = SQ \times RQ$.
*   **Segmentation Quality (SQ):** Average IoU of matched segments.
*   **Recognition Quality (RQ):** F1 Score (Detection accuracy).
*   It penalizes both missed objects (FN), false alarms (FP), and inaccurate masks (low IoU).

## Q4: How does Panoptic FPN resolve conflicts between Instance and Semantic branches?
**Answer:**
*   **Conflict:** A pixel might be predicted as "Person" by the Instance branch and "Road" by the Semantic branch.
*   **Resolution:** Usually, the **Instance branch takes priority**.
*   The instance masks are pasted first. Any remaining pixels are filled with the semantic prediction.

## Q5: What is MaskFormer?
**Answer:**
*   A Transformer-based architecture that unifies segmentation.
*   Instead of predicting a box and then a mask (Mask R-CNN), it predicts a set of binary masks directly using object queries.
*   Each mask is then classified.
*   It treats Semantic Segmentation as a special case where there is only one instance per class.

## Q6: Why is the Offset Map used in Panoptic-DeepLab?
**Answer:**
*   To group pixels into instances without using bounding boxes.
*   For every pixel belonging to a "Thing", the network predicts a 2D vector pointing to the center of that instance.
*   During inference, pixels pointing to the same center are grouped together.

## Q7: What is "Void" in Panoptic Segmentation?
**Answer:**
*   Pixels that do not belong to any known class or are too ambiguous to label.
*   In the PQ metric, predictions in the Void region are ignored (not counted as FP/FN).

## Q8: Can we use YOLO for Panoptic Segmentation?
**Answer:**
*   Standard YOLO is for detection.
*   **YOLOP (YOLO Panoptic):** Adds a semantic segmentation branch (for road/lane) to the YOLO backbone.
*   So yes, but it requires architectural modification similar to Panoptic FPN.

## Q9: Why is Panoptic Segmentation important for Self-Driving Cars?
**Answer:**
*   **Safety:** You need to detect obstacles (Things: Cars, Pedestrians).
*   **Navigation:** You need to know where the drivable surface is (Stuff: Road, Lane markings).
*   Panoptic provides both in a single coherent output, ensuring no pixel is left unexplained.

## Q10: Implement PQ calculation logic (conceptual).
**Answer:**
```python
def compute_pq(pred_segments, gt_segments):
    # Match segments based on IoU > 0.5
    matches = []
    for p in pred_segments:
        for g in gt_segments:
            if iou(p, g) > 0.5:
                matches.append((p, g))
                
    tp = len(matches)
    fp = len(pred_segments) - tp
    fn = len(gt_segments) - tp
    
    iou_sum = sum([iou(p, g) for p, g in matches])
    
    pq = iou_sum / (tp + 0.5 * fp + 0.5 * fn)
    return pq
```

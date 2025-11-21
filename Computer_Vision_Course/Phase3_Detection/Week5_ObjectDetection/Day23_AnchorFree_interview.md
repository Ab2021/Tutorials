# Day 23 Interview Questions: Anchor-Free Detection

## Q1: What are the main disadvantages of Anchor-based detectors?
**Answer:**
1.  **Hyperparameter Sensitivity:** Performance depends heavily on anchor sizes/ratios.
2.  **Imbalance:** Massive number of negative anchors slows down training.
3.  **Rigidity:** Fixed anchors struggle with objects of unusual shapes (e.g., very thin/tall).
4.  **Complexity:** IoU matching logic is complicated.

## Q2: How does FCOS define a "positive" sample?
**Answer:**
*   A location $(x, y)$ on the feature map is positive if it falls inside the ground truth bounding box of an object.
*   It also uses a "Center Sampling" trick: only pixels near the center of the box are positive, ignoring those near edges to improve quality.

## Q3: What is the "Centerness" branch in FCOS?
**Answer:**
*   A branch that predicts a score (0 to 1) indicating how close a pixel is to the center of the object.
*   During inference, the classification score is multiplied by the centerness score.
*   This suppresses low-quality bounding boxes generated from pixels far from the center, which tend to be inaccurate.

## Q4: How does CenterNet handle NMS?
**Answer:**
*   It effectively **removes** the need for standard NMS.
*   It finds peaks in the heatmap using a $3 \times 3$ Max Pooling.
*   If a pixel value is equal to the max-pooled value, it is a peak (local maximum).
*   This is much faster than sorting boxes and computing IoUs.

## Q5: Why does CenterNet need an "Offset" head?
**Answer:**
*   The output heatmap is usually downsampled (stride $R=4$).
*   Mapping a peak back to the original image introduces a quantization error of up to $R/2$ pixels.
*   The Offset head predicts a small correction $(\delta x, \delta y)$ to recover the precise sub-pixel location.

## Q6: Explain the Gaussian Kernel in CenterNet training.
**Answer:**
*   Instead of a hard 1/0 label (center vs background), CenterNet places a 2D Gaussian blob at the ground truth center.
*   This tells the network: "The exact center is best (1.0), but pixels nearby are also okay (0.8), and far away is background (0.0)."
*   This acts as a form of label smoothing and handles ambiguity.

## Q7: What is the difference between FCOS and RetinaNet?
**Answer:**
*   **RetinaNet:** Places anchors at every location. Classifies anchors. Regresses offsets from anchors.
*   **FCOS:** No anchors. Classifies pixels. Regresses distances to 4 sides directly from the pixel.
*   Both use FPN and Focal Loss. FCOS is essentially RetinaNet with anchors removed.

## Q8: How does CornerNet group corners?
**Answer:**
*   It detects all Top-Left corners and all Bottom-Right corners.
*   It predicts an **Embedding Vector** for each corner.
*   Loss function forces corners from the same object to have similar embeddings, and different objects to have different embeddings.
*   During inference, it matches TL and BR corners with small embedding distances.

## Q9: Why are Anchor-Free methods becoming popular?
**Answer:**
*   **Simplicity:** Easier to implement and debug.
*   **Generalization:** Less bias towards specific object shapes (defined by anchors).
*   **Dense Prediction:** Aligns detection with segmentation, enabling unified frameworks (like Mask R-CNN but anchor-free).

## Q10: Implement the Centerness target calculation.
**Answer:**
```python
def compute_centerness(l, t, r, b):
    # l, t, r, b are distances to the 4 sides
    # centerness = sqrt(min(l, r) / max(l, r) * min(t, b) / max(t, b))
    
    left_right = torch.min(l, r) / torch.max(l, r)
    top_bottom = torch.min(t, b) / torch.max(t, b)
    return torch.sqrt(left_right * top_bottom)
```

# Day 28 Interview Questions: Mask R-CNN

## Q1: Why is RoIPool bad for segmentation?
**Answer:**
*   RoIPool performs **quantization** (rounding coordinates to integers).
*   This introduces a misalignment between the extracted features and the original image pixels.
*   While classification is translation-invariant (doesn't care about small shifts), segmentation requires pixel-perfect alignment.
*   RoIAlign fixes this using bilinear interpolation.

## Q2: How does Mask R-CNN handle class competition in the mask head?
**Answer:**
*   It decouples mask prediction and class prediction.
*   The mask branch predicts $K$ binary masks (one for each class) independently using Sigmoid.
*   It does **not** use Softmax across classes.
*   The specific mask used for loss/inference is selected based on the class predicted by the classification branch.
*   This avoids competition between classes (e.g., "Person" vs "Rider").

## Q3: What is the loss function of Mask R-CNN?
**Answer:**
$$ L = L_{cls} + L_{box} + L_{mask} $$
*   $L_{cls}$: Cross-Entropy (RPN + Head).
*   $L_{box}$: Smooth L1 (RPN + Head).
*   $L_{mask}$: Binary Cross-Entropy (Average over pixels).

## Q4: Can Mask R-CNN detect objects without bounding boxes?
**Answer:**
**No.**
*   It is a two-stage detector.
*   It *must* generate region proposals (boxes) first.
*   The mask head operates on the features extracted from these boxes (via RoIAlign).
*   If the box is missing or bad, the mask will be missing or bad.

## Q5: Explain the difference between YOLACT and Mask R-CNN.
**Answer:**
*   **Mask R-CNN:** Two-stage. "Detect then Segment". Extracts local features for each object. Slow but accurate.
*   **YOLACT:** One-stage. "Prototype + Coefficients". Generates global prototype masks and combines them linearly. Fast (Real-time) but slightly less accurate on small details.

## Q6: How does RoIAlign use Bilinear Interpolation?
**Answer:**
*   If a sampling point $(x, y)$ falls between 4 grid points $(x_1, y_1), (x_2, y_2), \dots$
*   The value $V(x, y)$ is computed as a weighted average of the 4 neighbors, where weights depend on the distance.
*   This allows sampling feature values at floating-point coordinates.

## Q7: What is Panoptic Segmentation?
**Answer:**
*   A combination of Semantic Segmentation (Stuff: Sky, Road) and Instance Segmentation (Things: Car, Person).
*   Every pixel is assigned a (Class, Instance ID).
*   Mask R-CNN only does Instance (ignores "Stuff"). DeepLab only does Semantic (ignores "Instances"). Panoptic FPN combines them.

## Q8: Why does Mask R-CNN use FPN?
**Answer:**
*   To detect objects at multiple scales.
*   Small objects are detected on high-res feature maps (P2, P3).
*   Large objects are detected on low-res feature maps (P4, P5).
*   RoIAlign extracts features from the appropriate level based on the box size.

## Q9: What is the resolution of the predicted mask?
**Answer:**
*   The mask head typically outputs a $28 \times 28$ mask.
*   During inference, this small mask is resized (upsampled) to the size of the bounding box in the original image and then thresholded (0.5).

## Q10: Implement Bilinear Interpolation logic.
**Answer:**
```python
def bilinear_interpolate(img, x, y):
    # x, y are float coordinates
    x0 = int(x)
    x1 = x0 + 1
    y0 = int(y)
    y1 = y0 + 1
    
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)
    
    return wa*img[y0,x0] + wb*img[y1,x0] + wc*img[y0,x1] + wd*img[y1,x1]
```

# Day 3 Interview Questions: Edge Detection & Features

## Q1: Explain the Canny edge detector algorithm.
**Answer:**
Canny is a multi-stage optimal edge detector:

1. **Gaussian smoothing:** Reduce noise.
2. **Gradient computation:** Calculate magnitude and direction using Sobel.
3. **Non-maximum suppression:** Thin edges to single-pixel width by suppressing non-peak gradients.
4. **Double thresholding:** Classify as strong ($M > T_{high}$), weak ($T_{low} < M < T_{high}$), or suppressed.
5. **Edge tracking by hysteresis:** Keep strong edges + weak edges connected to strong edges.

**Why optimal?**
- **Good detection:** Low error rate.
- **Good localization:** Edges close to true edges.
- **Single response:** One detection per edge.

## Q2: What is non-maximum suppression and why is it important?
**Answer:**
**NMS** suppresses pixels that are not local maxima along the gradient direction.

**Algorithm:**
1. Quantize gradient direction to 0°, 45°, 90°, 135°.
2. Compare pixel with two neighbors along gradient direction.
3. Suppress if not the maximum.

**Importance:**
- **Thin edges:** Without NMS, edges are thick (multiple pixels).
- **Precise localization:** Single-pixel response at true edge location.
- **Reduces redundancy:** Fewer keypoints to process downstream.

**Used in:** Canny, SIFT, object detection (bounding box NMS).

## Q3: Compare Harris and Shi-Tomasi corner detectors.
**Answer:**
**Harris:**
$$ R = \det(M) - k \cdot \text{trace}(M)^2 = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2 $$
- $k \approx 0.04-0.06$ is a tunable parameter.
- **Problem:** $k$ is empirical, performance varies.

**Shi-Tomasi:**
$$ R = \min(\lambda_1, \lambda_2) $$
- **Advantage:** No empirical parameter $k$.
- **More stable:** Better corner selection.
- **Geometric meaning:** Minimum eigenvalue = worst-case gradient variation.

**When to use:** Shi-Tomasi is generally preferred (better stability, no parameter tuning).

## Q4: Why use Laplacian of Gaussian (LoG) instead of just Laplacian?
**Answer:**
**Laplacian alone:**
- Second derivative → very noise-sensitive.
- High-frequency noise gets amplified.

**Laplacian of Gaussian:**
$$ LoG = \nabla^2(G * I) = (\nabla^2 G) * I $$

**Advantages:**
1. **Smoothing first:** Gaussian removes noise before differentiation.
2. **Mathematically equivalent:** Can pre-compute $\nabla^2 G$ kernel.
3. **Scale-space:** Different $\sigma$ detect blobs at different scales.

**Trade-off:** More computation (larger kernel), but much more robust.

## Q5: What is the difference between edges and corners?
**Answer:**
**Edge:**
- High gradient in **one direction** only.
- Eigenvalues: One large ($\lambda_1 \gg \lambda_2$).
- **Examples:** Object boundaries, occlusion boundaries.

**Corner:**
- High gradient in **multiple directions** (edge intersection).
- Eigenvalues: Both large ($\lambda_1 \approx \lambda_2$, both >> 0).
- **Examples:** Building corners, "L" or "T" junctions.

**Why corners matter:** More distinctive than edges, better for matching/tracking.

## Q6: How does FAST corner detector work?
**Answer:**
**FAST** (Features from Accelerated Segment Test):

1. Consider 16 pixels on circle of radius 3 around point $p$.
2. Classify each pixel as:
   - Brighter: $I > I_p + T$
   - Similar: $|I - I_p| \leq T$
   - Darker: $I < I_p - T$
3. If $\geq N$ **contiguous** pixels are all brighter OR all darker → corner.
4. Typically $N=12$ (FAST-12).

**Speedup:** Use machine learning to build decision tree:
- Check pixels 1, 5, 9, 13 first (quick rejection).
- Only check all 16 if promising.

**Why fast:** Very few operations per pixel, optimized with decision tree.

## Q7: What is scale-space and why is it important?
**Answer:**
**Scale-space:** Family of images at different smoothing levels:
$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

**Why important:**
- **Objects at different sizes:** A small object at one scale may be large at another.
- **Consistent detection:** Detect features regardless of object size.
- **Example:** SIFT builds scale-space pyramid, detects keypoints at multiple scales.

**Gaussian uniqueness:** Only Gaussian scale-space satisfies causality (no new extrema as $\sigma$ increases).

## Q8: Implement Canny edge detection in Python.
**Answer:**
```python
import cv2
import numpy as np

# Method 1: OpenCV (recommended)
edges = cv2.Canny(image, threshold1=50, threshold2=150, 
                  apertureSize=3, L2gradient=True)

# Method 2: Manual (for understanding)
def canny_manual(img, low_thresh=50, high_thresh=150):
    # 1. Gaussian smoothing
    blurred = cv2.GaussianBlur(img, (5, 5), 1.4)
    
    # 2. Gradient computation
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(sobelx**2 + sobely**2)
    angle = np.arctan2(sobely, sobelx)
    
    # 3. Non-maximum suppression
    nms = non_max_suppression(magnitude, angle)
    
    # 4. Double thresholding
    strong = (nms > high_thresh)
    weak = (nms >= low_thresh) & (nms <= high_thresh)
    
    # 5. Edge tracking by hysteresis
    edges = hysteresis(strong, weak)
    
    return edges

# Parameters:
# threshold1: Low threshold for hysteresis
# threshold2: High threshold for hysteresis
# Typical ratio: high = 2-3 × low
```

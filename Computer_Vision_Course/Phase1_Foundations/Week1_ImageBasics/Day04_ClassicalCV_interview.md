# Day 4 Interview Questions: Classical CV Features

## Q1: Explain the SIFT algorithm in detail.
**Answer:**
**SIFT** (Scale-Invariant Feature Transform) has 4 stages:

1. **Scale-space extrema detection:**
   - Build DoG pyramid: $D(x,y,\sigma) = L(x,y,k\sigma) - L(x,y,\sigma)$
   - Detect local extrema (26 neighbors check)

2. **Keypoint localization:**
   - Sub-pixel refinement via quadratic fit
   - Reject low-contrast ($|D| < 0.03$) and edges (eigenvalue ratio)

3. **Orientation assignment:**
   - 36-bin histogram of gradient orientations
   - Peak = dominant orientation (multiple peaks → multiple keypoints)

4. **Descriptor:**
   - 16×16 patch around keypoint, rotated to orientation
   - 4×4 grid of cells, 8-bin histogram each
   - 128D descriptor (4×4×8), normalized

**Invariances:** Scale, rotation, partial illumination/viewpoint.

## Q2: Why is SURF faster than SIFT?
**Answer:**
**SURF optimizations:**

1. **Integral images:** Box filters computable in $O(1)$ time
   $$ \sum I(x,y) = II(x_2,y_2) - II(x_1,y_2) - II(x_2,y_1) + II(x_1,y_1) $$

2. **Fast-Hessian detector:** Approximate Hessian with box filters

3. **64D descriptor:** vs SIFT's 128D (2× smaller, faster matching)

4. **Simplified grid:** 4×4 instead of complex interpolation

**Speedup:** 3-5× faster while maintaining similar accuracy.

## Q3: What is HOG and why is it effective for pedestrian detection?
**Answer:**
**HOG** (Histogram of Oriented Gradients):

**Algorithm:**
1. Gradient computation (simple [-1,0,1] filters)
2. 8×8 pixel cells, 9 orientation bins
3. 2×2 cell blocks, L2-norm normalization
4. Concatenate all block descriptors

**Why effective:**
- **Captures shape:** Edge/gradient structure of human form
- **Illumination invariant:** Block normalization
- **Dense:** Overlapping blocks capture spatial information
- **Discriminative:** Human silhouette has distinctive gradient patterns

**Typical:** 64×128 window → 3780D descriptor fed to linear SVM.

## Q4: Explain Lowe's ratio test for feature matching.
**Answer:**
For each query descriptor, find 2 nearest neighbors with distances $d_1 < d_2$:

**Accept match if:**
$$ \frac{d_1}{d_2} < 0.75 $$

**Rationale:**
- **Good match:** Descriptor is distinctive, $d_1 \ll d_2$ (low ratio)
- **Ambiguous:** Multiple similar matches, $d_1 \approx d_2$ (high ratio)

**Effect:** Rejects ~90% of false matches while keeping ~95% of correct matches (empirically).

## Q5: Compare SIFT vs ORB. When to use each?
**Answer:**

**SIFT:**
- **Pros:** Very robust, scale/rotation invariant, 128D floating-point
- **Cons:** Slow, patented (until 2020), large descriptor
- **Use:** High accuracy needed, offline processing

**ORB:**
- **Pros:** Very fast, binary descriptor (Hamming distance), free
- **Cons:** Less rotation invariant than SIFT, less distinctive
- **Use:** Real-time applications (SLAM, AR), mobile devices

**Rule:** SIFT for accuracy, ORB for speed.

## Q6: What are binary descriptors and their advantages?
**Answer:**
**Binary descriptors** (BRIEF, ORB, BRISK, FREAK) use intensity comparisons:

$$ b_i = \begin{cases} 1 & \text{if } I(p_i) < I(q_i) \\ 0 & \text{otherwise} \end{cases} $$

**Advantages:**
1. **Fast computation:** Simple comparisons vs complex histograms
2. **Compact:** 256 bits = 32 bytes vs 128 floats = 512 bytes (SIFT)
3. **Fast matching:** Hamming distance via POPCNT (hardware accelerated)
4. **Memory efficient:** 16× smaller than SIFT

**Trade-off:** Less distinctive than SIFT, but acceptable for many applications.

## Q7: What is the affine transformation and how does ASIFT handle it?
**Answer:**
**Affine transformation:**
$$ \begin{bmatrix} x' \\ y' \end{bmatrix} = \begin{bmatrix} a & b \\ c & d \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} + \begin{bmatrix} t_x \\ t_y \end{bmatrix} $$

**SIFT limitations:** Not fully affine invariant (only rotation + uniform scale).

**ASIFT:** Simulate viewpoint changes:
- Warp image with different tilt/rotation
- Detect SIFT on each warped version
- Merge keypoints

**Coverage:** ~100 simulated views for full affine invariance.

**Trade-off:** 100× slower but handles extreme viewpoints.

## Q8: Implement feature matching with geometric verification.
**Answer:**
```python
import cv2
import numpy as np

# Detect and match
sift = cv2.SIFT_create()
kp1, desc1 = sift.detectAndCompute(img1, None)
kp2, desc2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

# Geometric verification with RANSAC
if len(good) >= 4:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)
    
    # Find homography
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    
    # Filter inliers
    inliers = [good[i] for i in range(len(good)) if mask[i]]
    
    print(f"Matches: {len(good)}, Inliers: {len(inliers)}")
```

**RANSAC parameters:**
- Method: `cv2.RANSAC` or `cv2.LMEDS`
- Threshold: 5.0 pixels (reprojection error)
- Confidence: 0.995 (default)

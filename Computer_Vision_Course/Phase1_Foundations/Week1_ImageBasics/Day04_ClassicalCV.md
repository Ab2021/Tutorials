# Day 4: Classical Computer Vision - SIFT, SURF, HOG

## 1. Scale-Invariant Feature Transform (SIFT)
**SIFT** (Lowe, 2004) detects and describes local features invariant to scale, rotation, and partially invariant to illumination and viewpoint.

### 1.1 Keypoint Detection (DoG Extrema)
Build scale-space using Difference of Gaussians:
$$ D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) $$

**Detect extrema** by comparing each pixel with 26 neighbors (8 in same scale + 9 in scale above + 9 in scale below).

### 1.2 Keypoint Localization
Fit 3D quadratic to refine location:
$$ \hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}} $$

**Reject if:**
- Low contrast: $|D(\hat{\mathbf{x}})| < 0.03$
- Edge response: $\frac{(\text{tr}(H))^2}{\det(H)} > \frac{(r+1)^2}{r}$ where $r=10$

### 1.3 Orientation Assignment
Create histogram of gradient orientations weighted by magnitude and Gaussian:

$$ h(\theta) = \sum_{(x,y) \in \text{region}} m(x,y) \cdot w(x,y) \cdot \mathbf{1}[\theta(x,y) \in \text{bin}] $$

where:
- $m(x,y) = \sqrt{I_x^2 + I_y^2}$: Gradient magnitude
- $w(x,y) = G_{\sigma}(x,y)$: Gaussian weight  
- 36 bins (10° each)

**Dominant orientation:** Peak in histogram. Multiple peaks → multiple keypoints at same location.

### 1.4 Descriptor Construction
16×16 patch around keypoint, rotated to dominant orientation.

Divide into 4×4 grid of cells. For each cell:
- Compute 8-bin orientation histogram
- Weight by magnitude and Gaussian

**Result:** 4×4×8 = **128-dimensional descriptor**.

**Normalization:**
1. Normalize to unit length.
2. Clip values to 0.2 (reduce illumination effects).
3. Renormalize.

## 2. Speeded-Up Robust Features (SURF)
**SURF** (Bay et al., 2006) is faster approximation of SIFT using integral images.

### 2.1 Fast-Hessian Detector
Use Hessian determinant for blob detection:
$$ \det(H) = D_{xx} D_{yy} - (0.9 D_{xy})^2 $$

**Approximate using box filters** (computable in constant time with integral images):
$$ \iint I(x,y) dx dy = II(x_2, y_2) - II(x_1, y_2) - II(x_2, y_1) + II(x_1, y_1) $$

### 2.2 Descriptor
64-dimensional (vs. SIFT's 128):
- 4×4 grid
- Each cell: 4 values (Σd_x, Σd_y, Σ|d_x|, Σ|d_y|)

**Speedup:** 3-5× faster than SIFT due to integral images and smaller descriptor.

## 3. Histogram of Oriented Gradients (HOG)
**HOG** (Dalal & Triggs, 2005) is a dense descriptor for object detection (especially pedestrians).

### 3.1 Algorithm
1. **Gradient computation:**
   $$ G_x = [-1, 0, 1], \quad G_y = [-1, 0, 1]^T $$
   
2. **Cell histograms:**
   - Divide image into 8×8 pixel cells
   - 9 orientation bins (unsigned: 0-180° or signed: 0-360°)
   - Weight votes by gradient magnitude

3. **Block normalization:**
   - Group cells into overlapping 2×2 blocks (16×16 pixels)
   - Normalize each block: $v' = \frac{v}{\sqrt{||v||^2 + \epsilon^2}}$
   
4. **Concatenate** all block descriptors

**For 64×128 detection window:**
- 7×15 blocks (with stride 8)
- Each block: 2×2×9 = 36 features
- **Total: 7×15×36 = 3780 dimensions**

### 3.2 Why HOG Works
- **Captures edge/gradient structure** (shape, appearance)
- **Local normalization** → illumination invariant
- **Overlapping blocks** → spatial continuity
- **Dense sampling** → captures fine details

## 4. Feature Matching

### Brute-Force Matcher
Compare each descriptor in set A with all in set B:
$$ d(f_i, f_j) = ||f_i - f_j||_2 \text{ (Euclidean)} $$

**Time:** $O(N \times M \times D)$ where $D$ is descriptor dimension.

### FLANN (Fast Library for Approximate Nearest Neighbors)
Use k-d trees or hierarchical k-means:
- **k-d tree:** Binary space partitioning
- **LSH:** Locality-sensitive hashing

**Time:** $O(\log N)$ average case (vs $O(N)$ brute force).

### Lowe's Ratio Test
Reject ambiguous matches. For each query descriptor:
1. Find 2 nearest neighbors: $d_1, d_2$ (distances to best and second-best)
2. Accept match only if: $\frac{d_1}{d_2} < 0.7-0.8$

**Rationale:** Good matches have distinctive descriptors (much closer to best than second-best).

## 5. Code Example: SIFT, SURF, HOG
```python
import cv2
import numpy as np

# SIFT (if available - patent expired 2020)
sift = cv2.SIFT_create()
keypoints_sift, descriptors_sift = sift.detectAndCompute(gray, None)

# ORB (free alternative to SIFT/SURF)
orb = cv2.ORB_create(nfeatures=500)
keypoints_orb, descriptors_orb = orb.detectAndCompute(gray, None)

# HOG for pedestrian detection
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
boxes, weights = hog.detectMultiScale(img, winStride=(8,8), padding=(4,4), scale=1.05)

# Feature matching
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

# Lowe's ratio test
good_matches = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good_matches.append(m)

# FLANN matcher (faster)
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches_flann = flann.knnMatch(desc1, desc2, k=2)
```

### Key Takeaways
- **SIFT:** Scale/rotation invariant, 128D descriptor, robust but slow.
- **SURF:** Faster approximation of SIFT using integral images, 64D descriptor.
- **HOG:** Dense descriptor for object detection, captures edge orientation statistics.
- **Feature matching:** Brute-force for accuracy, FLANN for speed, ratio test for reliability.

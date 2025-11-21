# Day 5 Interview Questions: Week 1 Review

## Q1: Compare different edge detection methods.
**Answer:**

| Method | Approach | Pros | Cons |
|--------|----------|------|------|
| **Sobel** | First derivative | Fast, simple | Sensitive to noise |
| **Canny** | Multi-stage | Optimal, thin edges | Parameter tuning needed |
| **LoG** | Second derivative | Scale-space | Expensive, double edges |
| **Scharr** | Optimized derivative | Better rotation invariance | Similar to Sobel |

**Canny advantages:**
1. Good detection (low error rate)
2. Good localization (edges close to true position)
3. Single response (no multiple edges)

## Q2: Explain the trade-off between SIFT and ORB.
**Answer:**

**SIFT:**
- **Accuracy:** High (128D float descriptor)
- **Speed:** Slow (~300ms for 1000 keypoints)
- **Memory:** 512 bytes per descriptor
- **Invariance:** Scale, rotation, partial affine

**ORB:**
- **Accuracy:** Moderate (256-bit binary)
- **Speed:** Fast (~30ms for 1000 keypoints)
- **Memory:** 32 bytes per descriptor
- **Invariance:** Scale, rotation

**Decision:** Use SIFT for offline/accuracy-critical tasks, ORB for real-time applications.

## Q3: What is scale-space and why is it important?
**Answer:**
**Scale-space:** Family of images at different scales:
$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

**Importance:**
1. **Scale invariance:** Detect features regardless of object size
2. **Hierarchical representation:** Coarse-to-fine processing
3. **Noise reduction:** Larger scales suppress noise

**Example:** SIFT uses DoG scale-space to find keypoints invariant to scale changes.

## Q4: How does RANSAC handle outliers?
**Answer:**
**Algorithm:**
1. Randomly sample minimal set (e.g., 4 points for homography)
2. Fit model to sample
3. Count inliers (points within threshold)
4. Repeat N iterations
5. Return model with most inliers

**Number of iterations:**
$$ N = \frac{\log(1 - p)}{\log(1 - (1-\epsilon)^s)} $$
- $p$: Desired probability of success (e.g., 0.99)
- $\epsilon$: Outlier ratio
- $s$: Sample size

**Example:** With 50% outliers, 4-point sample, 99% confidence: $N \approx 72$ iterations.

## Q5: Implement image pyramid construction.
**Answer:**
```python
def build_pyramid(image, n_levels=4, scale_factor=2):
    """Build Gaussian pyramid."""
    pyramid = [image]
    
    for i in range(1, n_levels):
        # Gaussian blur
        blurred = cv2.GaussianBlur(pyramid[i-1], (5, 5), 0)
        
        # Downsample
        h, w = blurred.shape[:2]
        downsampled = cv2.resize(blurred, (w // scale_factor, h // scale_factor))
        
        pyramid.append(downsampled)
    
    return pyramid

def build_laplacian_pyramid(gaussian_pyramid):
    """Build Laplacian pyramid for reconstruction."""
    laplacian = []
    
    for i in range(len(gaussian_pyramid) - 1):
        # Upsample next level
        h, w = gaussian_pyramid[i].shape[:2]
        upsampled = cv2.resize(gaussian_pyramid[i+1], (w, h))
        
        # Difference
        lap = cv2.subtract(gaussian_pyramid[i], upsampled)
        laplacian.append(lap)
    
    # Last level
    laplacian.append(gaussian_pyramid[-1])
    
    return laplacian
```

## Q6: What is the difference between correlation and convolution?
**Answer:**

**Convolution:**
$$ (I * K)(x,y) = \sum_{i,j} I(x-i, y-j) \cdot K(i,j) $$

**Correlation:**
$$ (I \star K)(x,y) = \sum_{i,j} I(x+i, y+j) \cdot K(i,j) $$

**Difference:** Kernel is flipped in convolution.

**Practical:** For symmetric kernels (Gaussian), they're identical. OpenCV's `filter2D` performs correlation.

## Q7: Explain HOG descriptor computation.
**Answer:**
**Steps:**
1. **Gradients:** Compute $I_x, I_y$ using [-1, 0, 1] filters
2. **Magnitude/Orientation:** 
   - $M = \sqrt{I_x^2 + I_y^2}$
   - $\theta = \arctan(I_y / I_x)$
3. **Cell histograms:** 8×8 pixels, 9 orientation bins (0-180°)
4. **Block normalization:** 2×2 cells, L2-norm
5. **Concatenate:** All normalized blocks

**Dimension:** For 64×128 window: $(7 \times 15) \times (2 \times 2) \times 9 = 3780$

## Q8: How to handle illumination changes in feature matching?
**Answer:**

**Techniques:**
1. **Descriptor normalization:** SIFT normalizes to unit length
2. **Gradient-based:** HOG, SIFT use gradients (less sensitive)
3. **Preprocessing:** Histogram equalization, CLAHE
4. **Color spaces:** Use illumination-invariant spaces (HSV, Lab)
5. **Ratio test:** Lowe's ratio test filters ambiguous matches

**Example:**
```python
# Preprocessing
lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0)
l_enhanced = clahe.apply(l)
enhanced = cv2.merge([l_enhanced, a, b])
enhanced_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

# Robust descriptor
sift = cv2.SIFT_create()
kp, desc = sift.detectAndCompute(enhanced_bgr, None)
```

## Q9: What is non-maximum suppression and why is it used?
**Answer:**
**Purpose:** Thin edges/responses to single-pixel width.

**Algorithm (Canny):**
1. Compute gradient magnitude and direction
2. Round direction to 0°, 45°, 90°, 135°
3. Compare each pixel with neighbors along gradient direction
4. Suppress if not local maximum

**Code:**
```python
def non_max_suppression(magnitude, direction):
    """NMS for edge thinning."""
    M, N = magnitude.shape
    suppressed = np.zeros((M, N), dtype=np.float32)
    
    # Quantize angles
    angle = direction * 180 / np.pi
    angle[angle < 0] += 180
    
    for i in range(1, M-1):
        for j in range(1, N-1):
            q = 255
            r = 255
            
            # 0 degrees
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = magnitude[i, j+1]
                r = magnitude[i, j-1]
            # 45 degrees
            elif 22.5 <= angle[i,j] < 67.5:
                q = magnitude[i+1, j-1]
                r = magnitude[i-1, j+1]
            # 90 degrees
            elif 67.5 <= angle[i,j] < 112.5:
                q = magnitude[i+1, j]
                r = magnitude[i-1, j]
            # 135 degrees
            elif 112.5 <= angle[i,j] < 157.5:
                q = magnitude[i-1, j-1]
                r = magnitude[i+1, j+1]
            
            if magnitude[i,j] >= q and magnitude[i,j] >= r:
                suppressed[i,j] = magnitude[i,j]
    
    return suppressed
```

## Q10: Compare brute-force vs FLANN matching.
**Answer:**

**Brute-Force:**
- **Complexity:** $O(N \times M)$
- **Accuracy:** Exact nearest neighbors
- **Use:** Small datasets (<1000 features)
- **Distance:** L1, L2, Hamming

**FLANN:**
- **Complexity:** $O(\log N)$ average
- **Accuracy:** Approximate (99%+ correct)
- **Use:** Large datasets (>1000 features)
- **Algorithms:** KD-tree (SIFT), LSH (binary)

**Example:**
```python
# Brute-force
bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
matches = bf.knnMatch(desc1, desc2, k=2)

# FLANN
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50)
flann = cv2.FlannBasedMatcher(index_params, search_params)
matches = flann.knnMatch(desc1, desc2, k=2)
```

**Trade-off:** FLANN is 10-100× faster with minimal accuracy loss.

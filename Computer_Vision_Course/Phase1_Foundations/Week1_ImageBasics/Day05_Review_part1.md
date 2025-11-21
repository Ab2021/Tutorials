# Day 5 Deep Dive: Advanced Image Analysis Techniques

## 1. Automatic Parameter Selection

### Otsu's Method for Thresholding
**Problem:** Find optimal threshold $t^*$ to separate foreground/background.

**Criterion:** Maximize between-class variance:
$$ \sigma_B^2(t) = \omega_0(t) \omega_1(t) [\mu_0(t) - \mu_1(t)]^2 $$

Where:
- $\omega_0, \omega_1$: Class probabilities
- $\mu_0, \mu_1$: Class means

```python
def otsu_threshold(image: np.ndarray) -> int:
    """Compute Otsu's threshold."""
    hist, bins = np.histogram(image.flatten(), bins=256, range=[0, 256])
    hist = hist.astype(float) / hist.sum()
    
    cumsum = np.cumsum(hist)
    cumsum_mean = np.cumsum(hist * np.arange(256))
    
    global_mean = cumsum_mean[-1]
    
    max_variance = 0
    threshold = 0
    
    for t in range(256):
        w0 = cumsum[t]
        w1 = 1 - w0
        
        if w0 == 0 or w1 == 0:
            continue
        
        mu0 = cumsum_mean[t] / w0
        mu1 = (global_mean - cumsum_mean[t]) / w1
        
        variance = w0 * w1 * (mu0 - mu1) ** 2
        
        if variance > max_variance:
            max_variance = variance
            threshold = t
    
    return threshold
```

### Adaptive Canny Thresholds
**Median-based selection:**
$$ T_{low} = \max(0, 0.66 \times \text{median}(I)) $$
$$ T_{high} = \min(255, 1.33 \times \text{median}(I)) $$

## 2. Scale-Space Theory

### Gaussian Scale Space
**Definition:**
$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

**Properties:**
1. **Causality:** No new structures created at coarser scales
2. **Scale invariance:** $L(x, y, t\sigma) = L(tx, ty, \sigma)$
3. **Semi-group:** $G_{\sigma_1} * G_{\sigma_2} = G_{\sqrt{\sigma_1^2 + \sigma_2^2}}$

### Laplacian of Gaussian (LoG)
**Blob detection:**
$$ \nabla^2 L = \frac{\partial^2 L}{\partial x^2} + \frac{\partial^2 L}{\partial y^2} $$

**Scale-normalized:**
$$ \nabla^2_{norm} L = \sigma^2 \nabla^2 L $$

**DoG approximation:**
$$ \nabla^2 G \approx \frac{G(k\sigma) - G(\sigma)}{(k-1)\sigma^2} $$

```python
def build_scale_space(image: np.ndarray, n_octaves=4, n_scales=5) -> List[List[np.ndarray]]:
    """Build Gaussian scale-space pyramid."""
    k = 2 ** (1.0 / n_scales)
    sigma = 1.6
    
    pyramid = []
    
    for octave in range(n_octaves):
        octave_images = []
        
        # Downsample for this octave
        if octave == 0:
            base = image.copy()
        else:
            base = cv2.resize(pyramid[octave-1][-1], 
                            (image.shape[1] // (2**octave), 
                             image.shape[0] // (2**octave)))
        
        for scale in range(n_scales + 3):
            sigma_scale = sigma * (k ** scale)
            blurred = cv2.GaussianBlur(base, (0, 0), sigma_scale)
            octave_images.append(blurred)
        
        pyramid.append(octave_images)
    
    return pyramid

def compute_dog_pyramid(gaussian_pyramid: List[List[np.ndarray]]) -> List[List[np.ndarray]]:
    """Compute Difference of Gaussians."""
    dog_pyramid = []
    
    for octave in gaussian_pyramid:
        dog_octave = []
        for i in range(len(octave) - 1):
            dog = cv2.subtract(octave[i+1], octave[i])
            dog_octave.append(dog)
        dog_pyramid.append(dog_octave)
    
    return dog_pyramid
```

## 3. Advanced Feature Matching

### RANSAC Variants

**Standard RANSAC:**
```python
def ransac_homography(src_pts, dst_pts, threshold=5.0, max_iters=2000):
    """RANSAC for homography estimation."""
    best_inliers = []
    best_H = None
    n_points = len(src_pts)
    
    for _ in range(max_iters):
        # Random sample
        indices = np.random.choice(n_points, 4, replace=False)
        sample_src = src_pts[indices]
        sample_dst = dst_pts[indices]
        
        # Compute homography
        H = cv2.getPerspectiveTransform(
            sample_src.astype(np.float32),
            sample_dst.astype(np.float32)
        )
        
        # Count inliers
        projected = cv2.perspectiveTransform(src_pts.reshape(-1,1,2), H)
        distances = np.linalg.norm(projected - dst_pts.reshape(-1,1,2), axis=2).flatten()
        inliers = np.where(distances < threshold)[0]
        
        if len(inliers) > len(best_inliers):
            best_inliers = inliers
            best_H = H
    
    # Refine with all inliers
    if len(best_inliers) >= 4:
        best_H, _ = cv2.findHomography(
            src_pts[best_inliers],
            dst_pts[best_inliers],
            method=0  # Least squares
        )
    
    return best_H, best_inliers
```

**PROSAC (Progressive Sample Consensus):**
Uses quality-sorted matches, faster convergence.

**LO-RANSAC (Locally Optimized RANSAC):**
Local optimization step after finding good model.

## 4. Descriptor Matching Strategies

### Cross-Check Matching
```python
def cross_check_matching(desc1, desc2, ratio=0.75):
    """Bidirectional matching with cross-check."""
    bf = cv2.BFMatcher()
    
    # Forward matches
    matches12 = bf.knnMatch(desc1, desc2, k=2)
    good12 = []
    for m, n in matches12:
        if m.distance < ratio * n.distance:
            good12.append(m)
    
    # Backward matches
    matches21 = bf.knnMatch(desc2, desc1, k=2)
    good21 = []
    for m, n in matches21:
        if m.distance < ratio * n.distance:
            good21.append(m)
    
    # Cross-check
    cross_checked = []
    for m12 in good12:
        for m21 in good21:
            if m12.queryIdx == m21.trainIdx and m12.trainIdx == m21.queryIdx:
                cross_checked.append(m12)
                break
    
    return cross_checked
```

### Spatial Verification
**Constraint:** Matched keypoints should have consistent spatial relationships.

```python
def spatial_verification(kp1, kp2, matches, max_distance=50):
    """Verify matches using spatial consistency."""
    if len(matches) < 3:
        return matches
    
    # Compute pairwise distances
    pts1 = np.array([kp1[m.queryIdx].pt for m in matches])
    pts2 = np.array([kp2[m.trainIdx].pt for m in matches])
    
    verified = []
    for i, m in enumerate(matches):
        # Check if spatial relationships are preserved
        dist1 = np.linalg.norm(pts1 - pts1[i], axis=1)
        dist2 = np.linalg.norm(pts2 - pts2[i], axis=1)
        
        # Ratio of distances should be similar
        ratio = dist2 / (dist1 + 1e-6)
        median_ratio = np.median(ratio[ratio > 0])
        
        if np.abs(ratio[i] - median_ratio) < 0.5:
            verified.append(m)
    
    return verified
```

## 5. Performance Optimization

### Integral Images for Fast Filtering
```python
class IntegralImage:
    """Efficient box filter computation."""
    
    def __init__(self, image: np.ndarray):
        self.integral = cv2.integral(image)
    
    def box_filter(self, x, y, width, height):
        """Compute sum in O(1) time."""
        x1, y1 = x, y
        x2, y2 = x + width, y + height
        
        A = self.integral[y1, x1]
        B = self.integral[y1, x2]
        C = self.integral[y2, x1]
        D = self.integral[y2, x2]
        
        return D - B - C + A
```

### GPU Acceleration with OpenCV
```python
# Upload to GPU
gpu_img = cv2.cuda_GpuMat()
gpu_img.upload(image)

# GPU SIFT (if available)
gpu_sift = cv2.cuda.SURF_CUDA_create(400)
keypoints, descriptors = gpu_sift.detectWithDescriptors(gpu_img, None)

# Download results
descriptors_cpu = descriptors.download()
```

## Summary
Advanced techniques enable robust, efficient image analysis through automatic parameter selection, scale-space theory, and optimized matching strategies.

# Day 3: Edge Detection & Feature Extraction

## 1. Image Gradients: Foundation of Edge Detection
An **edge** is a rapid change in intensity. The gradient captures this:

$$ \nabla I = \begin{bmatrix} \frac{\partial I}{\partial x} \\ \frac{\partial I}{\partial y} \end{bmatrix} = \begin{bmatrix} I_x \\ I_y \end{bmatrix} $$

**Gradient magnitude:**
$$ |\nabla I| = \sqrt{I_x^2 + I_y^2} $$

**Gradient direction:**
$$ \theta = \arctan\left(\frac{I_y}{I_x}\right) $$

**High gradient magnitude → Edge!**

## 2. Discrete Gradient Operators

### Roberts Cross
Fastest, simplest (2×2):
$$ G_x = \begin{bmatrix} +1 & 0 \\ 0 & -1 \end{bmatrix}, \quad G_y = \begin{bmatrix} 0 & +1 \\ -1 & 0 \end{bmatrix} $$

**Diagonal sensitive, very noise-prone.**

### Prewitt Operator
3×3 with averaging:
$$ G_x = \begin{bmatrix} -1 & 0 & +1 \\ -1 & 0 & +1 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -1 & -1 \\ 0 & 0 & 0 \\ +1 & +1 & +1 \end{bmatrix} $$

**Averages 3 pixels → reduces noise.**

### Sobel Operator
3×3 with weighted averaging (more weight to center):
$$ G_x = \begin{bmatrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{bmatrix}, \quad G_y = \begin{bmatrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{bmatrix} $$

**Most popular:** Good balance of performance and smoothing. **Separable:**
$$ G_x = \begin{bmatrix} 1 \\ 2 \\ 1 \end{bmatrix} \begin{bmatrix} -1 & 0 & +1 \end{bmatrix} $$

## 3. Canny Edge Detector
Multi-stage algorithm for optimal edge detection:

### Step 1: Gaussian Smoothing
Reduce noise:
$$ I_{smooth} = G_\sigma * I $$

### Step 2: Gradient Computation
Compute $I_x, I_y$ using Sobel:
$$ M = \sqrt{I_x^2 + I_y^2}, \quad \theta = \arctan(I_y / I_x) $$

### Step 3: Non-Maximum Suppression (NMS)
Thin edges to single-pixel width. For each pixel:
1. Quantize gradient direction $\theta$ to 0°, 45°, 90°, 135°.
2. Compare $M(x,y)$ with neighbors along gradient direction.
3. Suppress if not local maximum.

**Result:** Thin, well-localized edges.

### Step 4: Double Thresholding
Two thresholds: $T_{low}$ and $T_{high}$ (typically $T_{high} = 2 \cdot T_{low}$):
- **Strong edges:** $M > T_{high}$
- **Weak edges:** $T_{low} < M < T_{high}$  
- **Suppressed:** $M < T_{low}$

### Step 5: Edge Tracking by Hysteresis
- Keep all strong edges.
- Keep weak edges **connected to strong edges** (recursive).
- Discard other weak edges.

**Result:** Connected edges with reduced noise.

## 4. Laplacian Edge Detection
Second derivative operator:

$$ \nabla^2 I = \frac{\partial^2 I}{\partial x^2} + \frac{\partial^2 I}{\partial y^2} $$

**Discrete approximation:**
$$ \nabla^2 I \approx I(x+1,y) + I(x-1,y) + I(x,y+1) + I(x,y-1) - 4I(x,y) $$

**Kernel:**
$$ L = \begin{bmatrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{bmatrix} $$

**Edges:** Zero-crossings of Laplacian.

**Problem:** Very noise-sensitive (second derivative amplifies noise).

**Solution:** Laplacian of Gaussian (LoG):
$$ LoG = \nabla^2(G_\sigma * I) = (\nabla^2 G_\sigma) * I $$

## 5. Corner Detection: Harris Corner Detector
**Corners** are points where edges meet (high gradient variation in multiple directions).

### Structure Tensor (Second Moment Matrix)
$$ M = \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix} * G_\sigma $$

where $*G_\sigma$ denotes Gaussian weighting.

### Corner Response Function
$$ R = \det(M) - k \cdot \text{trace}(M)^2 $$
$$ R = \lambda_1 \lambda_2 - k(\lambda_1 + \lambda_2)^2 $$

where $\lambda_1, \lambda_2$ are eigenvalues of $M$, and $k \approx 0.04-0.06$.

**Interpretation:**
- **Both $\lambda_1, \lambda_2$ large:** Corner (high $R$).
- **One large, one small:** Edge (low $R$).
- **Both small:** Flat region (low $R$).

**Threshold:** Keep points where $R > T$ and perform NMS.

## 6. Shi-Tomasi Corner Detector
Improved version of Harris:

$$ R = \min(\lambda_1, \lambda_2) $$

**Advantage:** More stable selection of corners.

## 7. FAST (Features from Accelerated Segment Test)
Very fast corner detector:

1. Consider circle of 16 pixels around point $p$.
2. Point is a corner if:
   - $N$ **contiguous** pixels brighter than $I_p + T$, OR
   - $N$ contiguous pixels darker than $I_p - T$
3. Typically $N = 12$ (FAST-12).

**Speed:** Can use decision tree learned from training data.

## 8. Code Example: Edge & Corner Detection
```python
import cv2
import numpy as np

# Sobel edge detection
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)  # x-gradient
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)  # y-gradient
magnitude = np.sqrt(sobelx**2 + sobely**2)

# Canny edge detection
edges_canny = cv2.Canny(img, threshold1=50, threshold2=150)

# Harris corner detection
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
harris = cv2.cornerHarris(gray, blockSize=2, ksize=3, k=0.04)
corners_harris = (harris > 0.01 * harris.max())

# Shi-Tomasi corners
corners_shi = cv2.goodFeaturesToTrack(gray, maxCorners=100, 
                                       qualityLevel=0.01, minDistance=10)

# FAST corners
fast = cv2.FastFeatureDetector_create(threshold=20)
keypoints_fast = fast.detect(img, None)
```

### Key Takeaways
- **First derivative (gradient):** Detects edges via magnitude.
- **Second derivative (Laplacian):** Detects edges via zero-crossings (noise-sensitive).
- **Canny:** Multi-stage algorithm for optimal edge detection (smoothing, NMS, hysteresis).
- **Harris:** Detects corners using eigenvalues of structure tensor.
- **FAST:** Very fast corner detection for real-time applications.

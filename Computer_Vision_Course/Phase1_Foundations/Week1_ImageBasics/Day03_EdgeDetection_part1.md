# Day 3 Deep Dive: Multi-Scale Feature Detection

## 1. Scale-Space Extrema Detection
Objects appear at different scales. Need to detect features across scales.

### Image Pyramid
Progressively downsample image:
$$ L^{(s)} = \text{downsample}(L^{(s-1)}, \text{factor}=2) $$

**Levels:** Original, 1/2, 1/4, 1/8, ... resolution.

### Gaussian Pyramid
Smooth before downsampling to avoid aliasing:
$$ L^{(s)} = \text{downsample}(G_\sigma * L^{(s-1)}) $$

**Octave:** Group of scales with same resolution.

## 2. Difference of Gaussians (DoG)
Efficient approximation of scale-normalized Laplacian:

$$ D(x, y, \sigma) = L(x, y, k\sigma) - L(x, y, \sigma) $$

where $k$ is scale factor (typically $\sqrt{2}$).

**Scale-normalized Laplacian:**
$$ \sigma^2 \nabla^2 G $$

**DoG approximates this:**
$$ D \approx (k-1)\sigma^2 \nabla^2 G $$

**Used in SIFT** for keypoint detection.

## 3. Non-Maximum Suppression (NMS)
Suppress non-peak responses to get precise localization.

### Spatial NMS
For each pixel, suppress if not maximum in neighborhood:
```python
def nms_2d(response, window_size=3):
    maxpool = maximum_filter(response, size=window_size)
    peaks = (response == maxpool) & (response > threshold)
    return peaks
```

### Scale-Space NMS
Check 26 neighbors (8 spatial + 9×2 in adjacent scales):
- Current scale: 8 neighbors
- Scale below: 9 neighbors  
- Scale above: 9 neighbors

**Total: 26 comparisons per candidate.**

## 4. Sub-Pixel Localization
Fit 3D quadratic to refine keypoint location:

$$ D(\mathbf{x}) \approx D + \frac{\partial D^T}{\partial \mathbf{x}} \mathbf{x} + \frac{1}{2} \mathbf{x}^T \frac{\partial^2 D}{\partial \mathbf{x}^2} \mathbf{x} $$

**Extremum location:**
$$ \hat{\mathbf{x}} = -\frac{\partial^2 D}{\partial \mathbf{x}^2}^{-1} \frac{\partial D}{\partial \mathbf{x}} $$

**Reject if:**
- $|\hat{\mathbf{x}}| > 0.5$ (too far from discrete location)
- $|D(\hat{\mathbf{x}})| < 0.03$ (low contrast)

## 5. Edge Response Elimination
Harris uses eigenvalue ratio. In SIFT:

$$ \frac{(r+1)^2}{r} < \frac{(\lambda_1 + \lambda_2)^2}{\lambda_1 \lambda_2} $$

where $r = \lambda_1 / \lambda_2$ (typically reject if $r > 10$).

**Reject edge-like responses** (one eigenvalue >> other).

## 6. Hessian Matrix for Blob Detection
For blob detection, use Hessian determinant:

$$ H = \begin{bmatrix} I_{xx} & I_{xy} \\ I_{xy} & I_{yy} \end{bmatrix} $$

**Blob response:**
$$ \det(H) = I_{xx} I_{yy} - I_{xy}^2 $$

**Used in SURF** (Speeded-Up Robust Features).

## 7. Oriented FAST and Rotated BRIEF (ORB)
Modern efficient alternative to SIFT/SURF:

### Orientation Assignment
Use intensity centroid:
$$ m_{pq} = \sum_{x,y} x^p y^q I(x,y) $$

**Centroid:**
$$ C = \left( \frac{m_{10}}{m_{00}}, \frac{m_{01}}{m_{00}} \right) $$

**Orientation:**
$$ \theta = \arctan(m_{01}/m_{10}) $$

### Rotation Invariance
Rotate descriptor according to keypoint orientation.

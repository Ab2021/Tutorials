# Day 4 Deep Dive: Descriptor Design & Matching

## 1. Invariance Properties

### Scale Invariance
Detect at multiple scales in scale-space pyramid:
$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

**Octave:** Double $\sigma$ while halving resolution.
**Scales per octave:** Typically 3-5.

### Rotation Invariance
Assign canonical orientation based on dominant gradient direction:
$$ \theta_{dominant} = \arg\max_\theta h(\theta) $$

Rotate patch and descriptor to this orientation.

### Illumination Invariance
**Gradient-based descriptors** are more robust to illumination than raw pixels:
- Additive change: $I' = I + c$ → $\nabla I' = \nabla I$ (invariant!)
- Multiplicative: $I' = a \cdot I$ → $\nabla I' = a \cdot \nabla I$ (normalized away)

**Normalization:** Unit-length descriptor handles multiplicative changes.

## 2. Binary Descriptors

### BRIEF (Binary Robust Independent Elementary Features)
Compare pixel pairs:
$$ f(\mathbf{p}; x, y) = \begin{cases} 1 & \text{if } I(x) < I(y) \\ 0 & \text{otherwise} \end{cases} $$

**Descriptor:** Concatenate $n$ binary tests (typically 128256 bits).

**Distance:** Hamming distance (count differing bits, very fast with POPCNT instruction).

### ORB (Oriented FAST and Rotated BRIEF)
- Keypoints: FAST corners
- Orientation: Intensity centroid
- Descriptor: Steered BRIEF (rotated according to orientation)

**Learning:** Select binary tests with low correlation and high variance.

### BRISK, FREAK
Improvements on BRIEF:
- **BRISK:** Specific sampling pattern
- **FREAK:** Retina-inspired sampling (dense at center, sparse at periphery)

## 3. Affine Invariance

### Affine-SIFT (ASIFT)
Simulate viewpoint changes:
$$ A = \begin{bmatrix} a & b \\ c & d \end{bmatrix} $$

Generate multiple affinely warped versions, detect SIFT on each.

**Coverage:** Latitude: 5-6 values, Longitude: rotate fully.

**Trade-off:** Computational cost vs. robustness.

## 4. Matching Strategies

### Cross-Check
Match A→B and B→A, keep only consistent matches:
$$ (i, j) \text{ is mutual best match} \iff \arg\min_k d(A_i, B_k) = j \land \arg\min_k d(B_j, A_k) = i $$

### RANSAC for Geometric Verification
Estimate homography/fundamental matrix, reject outliers:

1. Randomly select minimal set (4 points for homography)
2. Fit model
3. Count inliers (distance < threshold)
4. Repeat, keep best model

**Matches filtered by geometric consistency.**

## 5. Descriptor Dimensionality Reduction

### PCA on SIFT
Reduce 128D SIFT to 64D or 32D:
$$ f' = W^T (f - \mu) $$

where $W$ contains top-k eigenvectors.

**Trade-off:** Smaller, faster matching vs. slight accuracy loss.

### Product Quantization
Decompose descriptor into sub-vectors, quantize separately:
$$ f = [f_1, f_2, ..., f_m] $$

Each $f_i$ quantized to nearest centroid in its codebook.

**Asymmetric distance:** Fast approximate nearest neighbor search.

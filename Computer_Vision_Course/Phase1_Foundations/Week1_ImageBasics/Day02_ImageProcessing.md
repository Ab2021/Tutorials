# Day 2: Image Processing Operations

## 1. Convolution: The Fundamental Operation
**Convolution** is the core operation in image processing and deep learning:

$$ (I * K)(x, y) = \sum_{m}\sum_{n} I(x-m, y-n) \cdot K(m, n) $$

where:
- $I$: Input image
- $K$: Kernel (filter, mask)
- $*$: Convolution operator

**Discrete 2D convolution:**
$$ G[i,j] = \sum_{m=-k}^{k} \sum_{n=-k}^{k} I[i+m, j+n] \cdot K[m, n] $$

### Properties
- **Commutative:** $I * K = K * I$
- **Associative:** $(I * K_1) * K_2 = I * (K_1 * K_2)$ (can cascade filters)
- **Linear:** $I * (aK_1 + bK_2) = a(I * K_1) + b(I * K_2)$

### Correlation vs. Convolution
- **Correlation:** $K$ not flipped: $(I \star K)(x,y) = \sum_m\sum_n I(x+m, y+n) \cdot K(m,n)$
- **Convolution:** $K$ flipped: $(I * K)(x,y) = \sum_m\sum_n I(x-m, y-n) \cdot K(m,n)$

In deep learning, we typically use **correlation** but call it "convolution."

## 2. Common Filters & Kernels

### Box Filter (Mean Filter)
$$ K_{box} = \frac{1}{9} \begin{bmatrix} 1 & 1 & 1 \\ 1 & 1 & 1 \\ 1 & 1 & 1 \end{bmatrix} $$

**Effect:** Blur (average neighborhood). Reduces noise but also reduces sharpness.

### Gaussian Filter
$$ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2 + y^2}{2\sigma^2}} $$

**Discrete 5×5 Gaussian ($\sigma = 1$):**
$$ K_{gauss} = \frac{1}{273} \begin{bmatrix} 
1 & 4 & 7 & 4 & 1 \\
4 & 16 & 26 & 16 & 4 \\
7 & 26 & 41 & 26 & 7 \\
4 & 16 & 26 & 16 & 4 \\
1 & 4 & 7 & 4 & 1
\end{bmatrix} $$

**Properties:**
- **Separable:** $G(x,y) = G_x(x) \cdot G_y(y)$ (can apply 1D filters separately, $O(n^2)$ → $O(2n)$)
- **Smooth:** Better at preserving edges than box filter.
- **Isotropic:** Rotationally symmetric.

### Sharpening Filter
$$ K_{sharpen} = \begin{bmatrix} 0 & -1 & 0 \\ -1 & 5 & -1 \\ 0 & -1 & 0 \end{bmatrix} $$

**Derivation:** $I_{sharp} = I + \alpha \cdot \nabla^2 I$ (add Laplacian to enhance edges).

## 3. Bilateral Filter
Preserves edges while smoothing:

$$ BF[I]_p = \frac{1}{W_p} \sum_{q \in S} G_{\sigma_s}(||p-q||) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q $$

where:
- $G_{\sigma_s}$: Spatial Gaussian (distance-based weight)
- $G_{\sigma_r}$: Range Gaussian (intensity difference weight)
- $W_p$: Normalization factor

**Key idea:** Weight neighbors by both **spatial proximity** AND **intensity similarity**.
- Similar intensities (likely same region) → High weight.
- Different intensities (likely edge) → Low weight → Edge preserved.

## 4. Morphological Operations
Binary image operations based on set theory:

### Erosion
$$ (I \ominus B)(x, y) = \min_{(s,t) \in B} I(x+s, y+t) $$
**Effect:** Shrink foreground, remove small objects, break connections.

### Dilation
$$ (I \oplus B)(x, y) = \max_{(s,t) \in B} I(x+s, y+t) $$
**Effect:** Expand foreground, fill small holes, connect components.

### Opening
$$ I \circ B = (I \ominus B) \oplus B $$
**Effect:** Remove small bright spots, smooth boundaries (erosion then dilation).

### Closing
$$ I \bullet B = (I \oplus B) \ominus B $$
**Effect:** Fill small dark spots, close gaps (dilation then erosion).

## 5. Histogram Operations

### Histogram Equalization
Enhance contrast by spreading out intensity distribution:

1. Compute CDF:
$$ CDF(i) = \sum_{j=0}^{i} h(j) $$

2. Map intensities:
$$ I'(x,y) = \text{round}\left( \frac{CDF(I(x,y)) - CDF_{min}}{MN - CDF_{min}} \times 255 \right) $$

**Result:** Uniform histogram → Maximum contrast.

### Adaptive Histogram Equalization (CLAHE)
- Divide image into tiles.
- Equalize each  tile separately.
- **Clip limit:** Prevent over-amplification of noise.

## 6. Frequency Domain Filtering

### Fourier Transform
$$ F(u, v) = \sum_{x=0}^{M-1} \sum_{y=0}^{N-1} I(x,y) e^{-i2\pi(ux/M + vy/N)} $$

**Convolution Theorem:**
$$ I * K = \mathcal{F}^{-1}\{ \mathcal{F}(I) \cdot \mathcal{F}(K) \} $$

**Spatial convolution = Frequency multiplication!**

**For large kernels:** FFT-based convolution is faster (O(n² log n) vs O(n²k²)).

### Ideal Low-Pass Filter
$$ H(u, v) = \begin{cases} 
1 & \text{if } \sqrt{u^2 + v^2} \leq D_0 \\
0 & \text{otherwise}
\end{cases} $$

**Effect:** Blur (remove high frequencies = edges/details).

### Ideal High-Pass Filter
$$ H(u, v) = 1 - H_{LP}(u, v) $$

**Effect:** Sharpen (remove low frequencies = smooth regions).

## 7. Code Example: Implementing Filters
```python
import cv2
import numpy as np
from scipy.ndimage import convolve

# Gaussian blur
img_blur = cv2.GaussianBlur(img, ksize=(5,5), sigmaX=1.0)

# Custom convolution
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])  # Sharpen
img_sharp = convolve(img, kernel)

# Bilateral filter  
img_bilateral = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)

# Morphology
kernel_morph = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
img_opened = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel_morph)
img_closed = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel_morph)

# Histogram equalization
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_eq = cv2.equalizeHist(img_gray)

# CLAHE
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
img_clahe = clahe.apply(img_gray)
```

### Key Takeaways
- Convolution is the fundamental operation in image processing and CNNs.
- Gaussian blur is smoother and more edge-preserving than box blur (separable, too!).
- Bilateral filter preserves edges via range weighting.
- Morphological operations manipulate shapes in binary images.
- Histogram equalization maximizes contrast but can amplify noise.

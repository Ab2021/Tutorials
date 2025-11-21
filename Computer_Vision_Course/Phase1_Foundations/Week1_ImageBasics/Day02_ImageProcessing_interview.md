# Day 2 Interview Questions: Image Processing

## Q1: Explain the difference between convolution and correlation.
**Answer:**
**Convolution:** Kernel is **flipped** before sliding:
$$ (I * K)(x,y) = \sum_m \sum_n I(x-m, y-n) K(m,n) $$

**Correlation:** Kernel is **not flipped**:
$$ (I \star K)(x,y) = \sum_m \sum_n I(x+m, y+n) K(m,n) $$

**In practice:** Deep learning frameworks use correlation but call it "convolution." For symmetric kernels (Gaussian), they're identical.

## Q2: Why is the Gaussian filter better than a box filter for smoothing?
**Answer:**
**Gaussian advantages:**
1. **Smoother:** No sharp cutoffs, gradual weighting.
2. **Isotropic:** Rotationally symmetric (same in all directions).
3. **Separable:** Can compute as two 1D convolutions (faster: $O(n^2 \cdot 2k)$ vs $O(n^2 \cdot k^2)$).
4. **Unique:** Only linear filter that doesn't create new extrema (scale-space theory).
5. **Frequency domain:** Clean low-pass (Fourier transform of Gaussian is Gaussian).

**Box filter:** Ringing artifacts in frequency domain, directional bias on diagonals.

## Q3: What is the bilateral filter and when would you use it?
**Answer:**
**Bilateral filter** smooths while **preserving edges** by weighting pixels by both spatial and intensity similarity:

$$ BF[I]_p = \frac{1}{W_p} \sum_q G_{\sigma_s}(||p-q||) \cdot G_{\sigma_r}(|I_p - I_q|) \cdot I_q $$

- $G_{\sigma_s}$: Spatial Gaussian (nearby pixels weighted more)
- $G_{\sigma_r}$: Range Gaussian (similar intensities weighted more)

**Use cases:**
- Denoising while preserving edges
- Smoothing for visualization
- HDR tone mapping
- Texture removal

**Trade-off:** Slower than Gaussian ($O(N \cdot k^2)$), not separable.

## Q4: Explain morphological opening and closing. When do you use each?
**Answer:**
**Opening:** Erosion then dilation: $I \circ B = (I \ominus B) \oplus B$
- **Removes:** Small bright objects, noise spots
- **Smooths:** Object boundaries (from outside)
- **Use:** Remove text from images, clean up small artifacts

**Closing:** Dilation then erosion: $I \bullet B = (I \oplus B) \ominus B$
- **Fills:** Small dark holes, gaps, cracks
- **Smooths:** Object boundaries (from inside)
- **Connects:** Nearly touching objects
- **Use:** Fill missing parts, connect broken lines

**Rule of thumb:** Opening removes bright, closing removes dark.

## Q5: What is histogram equalization and what are its limitations?
**Answer:**
**Histogram equalization** spreads out intensity values to maximize contrast:

1. Compute CDF: $CDF(i) = \sum_{j \leq i} h(j)$
2. Map: $I'(x,y) = \frac{CDF(I(x,y)) - CDF_{min}}{M \cdot N - CDF_{min}} \times 255$

**Result:** Approximately uniform histogram.

**Limitations:**
1. **Noise amplification:** Noise gets enhanced too.
2. **Unnatural:** Can create artifacts in smooth regions.
3. **Global:** Doesn't adapt to local variations.
4. **Context-agnostic:** May not preserve relative intensities.

**Solution:** CLAHE (Adaptive) - equalize tiles separately with clip limit.

## Q6: Why is the convolution theorem useful?
**Answer:**
**Convolution theorem:**
$$ I * K = \mathcal{F}^{-1}\{ \mathcal{F}(I) \cdot \mathcal{F}(K) \} $$

**Spatial convolution = Frequency multiplication**

**Advantages:**
1. **Large kernels:** FFT-based convolution is $O(N \log N)$ vs $O(N \cdot k^2)$ for spatial.
2. **Understanding:** Visualize filter effects in frequency domain.
3. **Filter design:** Design filters by specifying frequency response.

**When to use FFT:**
- Kernel size > ~15×15
- Same filter applied many times (compute FFT once)

## Q7: What makes a filter separable and why does it matter?
**Answer:**
A 2D filter is **separable** if:
$$ K(x, y) = K_x(x) \cdot K_y(y) $$

**Computational savings:**
- **2D:** $(M \times N) * (k \times k)$ kernel = $O(MNk^2)$ operations
- **Separable:** $(M \times N) * (k \times 1)$ then $(M \times N) * (1 \times k)$ = $O(2MNk)$ operations

**Example:** 9×9 Gaussian:
- Non-separable: 81 multiplications per pixel
- Separable: 18 multiplications per pixel (4.5× speedup)

**Common separable filters:** Gaussian, Sobel, Box, Binomial.

## Q8: Implement Gaussian blur in Python.
**Answer:**
```python
import cv2
import numpy as np
from scipy.ndimage import gaussian_filter

# Method 1: OpenCV (fastest)
img_blur = cv2.GaussianBlur(img, ksize=(5, 5), sigmaX=1.0)

# Method 2: SciPy
img_blur = gaussian_filter(img, sigma=1.0)

# Method 3: Manual (for understanding)
def create_gaussian_kernel(size, sigma):
    kernel = np.fromfunction(
        lambda x, y: (1/(2*np.pi*sigma**2)) * 
        np.exp(-((x-(size-1)/2)**2 + (y-(size-1)/2)**2)/(2*sigma**2)),
        (size, size)
    )
    return kernel / np.sum(kernel)  # Normalize

kernel = create_gaussian_kernel(5, 1.0)
img_blur = cv2.filter2D(img, -1, kernel)
```

**Parameters:**
- `ksize`: Kernel size (odd numbers: 3, 5, 7...)
- `sigma`: Standard deviation (larger = more blur)
- Rule: `ksize ≈ 6*sigma` to capture 99% of Gaussian

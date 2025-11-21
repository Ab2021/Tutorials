# Day 2 Deep Dive: Advanced Filtering Theory

## 1. Filter Design Principles

### Separability
A 2D filter is **separable** if it can be decomposed into two 1D filters:
$$ K_{2D}(x, y) = K_x(x) \cdot K_y(y) $$

**Computational advantage:**
- **2D convolution:** $O(n^2 \cdot k^2)$ operations
- **Separable:** $O(n^2 \cdot 2k)$ operations

**Example - Gaussian:**
$$ G(x, y) = \frac{1}{2\pi\sigma^2} e^{-\frac{x^2+y^2}{2\sigma^2}} = \underbrace{\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{x^2}{2\sigma^2}}}_{G_x(x)} \cdot \underbrace{\frac{1}{\sqrt{2\pi}\sigma} e^{-\frac{y^2}{2\sigma^2}}}_{G_y(y)} $$

### Linearity and Superposition
Linear filters satisfy:
$$ L(a \cdot I_1 + b \cdot I_2) = a \cdot L(I_1) + b \cdot L(I_2) $$

**Non-linear filters:** Median filter, bilateral filter, morphological operations.

## 2. Scale-Space Theory

### Gaussian Scale-Space
A family of increasingly blurred images:
$$ L(x, y, \sigma) = G(x, y, \sigma) * I(x, y) $$

**Properties:**
- No new extrema created as $\sigma$ increases (causality).
- Linear scale-space kernel is unique: **Gaussian**.

### Laplacian of Gaussian (LoG)
$$ \nabla^2 G(x, y, \sigma) = \frac{1}{\pi\sigma^4} \left( \frac{x^2 + y^2}{\sigma^2} - 2 \right) e^{-\frac{x^2+y^2}{2\sigma^2}} $$

**Used for:** Blob detection (zero-crossings indicate blob locations).

### Difference of Gaussians (DoG)
Approximates LoG more efficiently:
$$ DoG(x, y, \sigma) = G(x, y, k\sigma) - G(x, y, \sigma) \approx \sigma \nabla^2 G $$

**SIFT uses DoG** for keypoint detection across scales.

## 3. Anisotropic Diffusion
Perona-Malik diffusion preserves edges while smoothing:

$$ \frac{\partial I}{\partial t} = \text{div}(c(|\nabla I|) \nabla I) $$

where $c(|\nabla I|)$ is a **diffusion coefficient** that decreases at edges:
$$ c(|\nabla I|) = e^{-(|\nabla I|/K)^2} $$

**Effect:** Strong smoothing in homogeneous regions, little smoothing across edges.

## 4. Wiener Filtering
Optimal filter for additive noise (MMSE criterion):

$$ H(u, v) = \frac{H^*(u, v)}{|H(u, v)|^2 + S_n(u,v)/S_I(u,v)} $$

where:
- $H$: Degradation function
- $S_n$: Noise power spectrum
- $S_I$: Signal power spectrum

**Balances:** Inverse filtering (restore signal) vs. noise amplification.

## 5. Non-Local Means Denoising
Use all similar patches in the image:

$$ NL[I](x) = \frac{1}{Z(x)} \sum_{y \in \Omega} w(x, y) I(y) $$

where weight depends on **patch similarity**:
$$ w(x, y) = e^{-\frac{||P(x) - P(y)||^2}{h^2}} $$

**Key insight:** Similar patches likely represent the same structure → average them to reduce noise while preserving details.

## 6. Guided Filter
Fast edge-preserving filter (linear complexity):

$$ q_i = \sum_j W_{ij}(I) p_j $$

where weights are derived from a **guidance image** $I$.

**Advantages over bilateral:**
- $O(N)$ complexity (bilateral is $O(N \cdot k^2)$)
- No gradient reversal artifacts
- Can use different image for guidance

## 7. Total Variation Denoising
Minimize total variation while staying close to observed image:

$$ \min_u \left\{ \int |\nabla u| dx + \frac{\lambda}{2} \int (u - I)^2 dx \right\} $$

**Effect:** Piecewise constant regions (cartoon-like), edges preserved.

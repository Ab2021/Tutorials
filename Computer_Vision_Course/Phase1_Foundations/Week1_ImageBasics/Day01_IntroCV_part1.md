# Day 1 Deep Dive: Image Formation & Sampling Theory

## 1. Pinhole Camera Model
The fundamental model of image formation:

### Geometry
$$ \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{f}{Z} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} $$

where:
- $(X, Y, Z)$: 3D point in world coordinates
- $(u, v)$: 2D image coordinates
- $f$: focal length
- $Z$: depth

**Perspective projection:** Distant objects appear smaller.

## 2. Camera Intrinsic Matrix
$$ K = \begin{bmatrix} f_x & s & c_x \\ 0 & f_y & c_y \\ 0 & 0 & 1 \end{bmatrix} $$

- $f_x, f_y$: Focal lengths in pixel units
- $c_x, c_y$: Principal point (optical center)
- $s$: Skew coefficient (usually 0)

**Full projection:**
$$ \mathbf{p} = K [R | t] \mathbf{P} $$
where $[R|t]$ is extrinsic matrix (rotation + translation).

## 3. Sampling & Aliasing

### Nyquist-Shannon Sampling Theorem
To perfectly reconstruct a signal, **sample at twice the highest frequency**:
$$ f_s \geq 2f_{max} $$

**In images:** High-frequency details (edges, textures) can cause aliasing if resolution is too low.

### Aliasing Examples
- **Moiré patterns:** Interference from regular textures.
- **Jagged edges:** Diagonal lines appear as staircases.
- **Temporal aliasing:** Wagon wheels appearing to spin backward in videos.

**Solution:** Anti-aliasing via **pre-filtering** (blur before downsampling).

## 4. Image Resolution & Information Content

### Spatial Frequency
Images can be decomposed into sinusoidal components (Fourier analysis):
$$ I(x, y) = \sum_{u,v} A(u,v) e^{i2\pi(ux + vy)} $$

- Low frequencies: Smooth regions, overall structure.
- High frequencies: Edges, textures, fine details.

### Information Theory
**Shannon entropy** quantifies information content:
$$ H(I) = -\sum_{i=0}^{255} p(i) \log_2 p(i) $$

- Uniform histogram → High entropy (random).
- Narrow histogram → Low entropy (predictable).

## 5. Radiometry & Photometry

### Radiance
Light energy per unit area per solid angle:
$$ L = \frac{d^2\Phi}{dA \cdot d\Omega \cdot \cos\theta} $$

### Image Irradiance Equation
$$ E(x, y) = \frac{\pi}{4} \left( \frac{d}{f} \right)^2 L \cos^4\alpha $$

where:
- $E$: Irradiance (energy received by sensor)
- $d$: Aperture diameter
- $\alpha$: Angle from optical axis

**Vignetting:** Brightness decreases toward image corners ($\cos^4\alpha$ term).

## 6. Sensor Considerations

### Dynamic Range
Ratio between brightest and darkest detectable intensities:
$$ DR = 20 \log_{10} \left( \frac{I_{max}}{I_{min}} \right) \text{ dB} $$

- **Low DR (8-bit):** ~48 dB (consumer cameras).
- **High DR (HDR):** > 100 dB (human eye ~100-150 dB).

### Pixel Size vs. Diffraction Limit
**Diffraction limit:** Fundamental resolution limit due to wave nature of light:
$$ \Delta x = 1.22 \lambda \frac{f}{D} $$

If pixels are smaller than this, **no additional detail captured**.

## 7. Color Sensitivity & Bayer Filters
Most cameras use **Bayer pattern:**
```
G  R  G  R
B  G  B  G
G  R  G  R
B  G  B  G
```
- 50% green (human eye most sensitive)
- 25% red, 25% blue

**Demosaicing:** Interpolation to recover full RGB at each pixel.

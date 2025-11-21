# Day 1 Interview Questions: CV Fundamentals

## Q1: Explain how a digital image is represented mathematically.
**Answer:**
A digital image is a discrete sampling of a continuous 2D light intensity function:
- **Grayscale:** $I: \mathbb{Z}^2 \rightarrow [0, 255]$, represented as an $M \times N$ matrix.
- **Color (RGB):** $I: \mathbb{Z}^2 \rightarrow [0, 255]^3$, represented as an $M \times N \times 3$ tensor.

Each pixel stores quantized intensity values. The image is:
- **Spatially discrete:** Finite pixel grid.
- **Intensity discrete:** Quantized to 8-bit (256 levels) typically.

## Q2: Why do we use different color spaces (RGB, HSV, LAB)?
**Answer:**
Each color space has advantages for different tasks:

**RGB:**
- **Pro:** Direct sensor output, natural for display.
- **Con:** Channels are correlated (changing brightness affects all channels).
- **Use:** General purpose, deep learning.

**HSV:**
- **Pro:** Separates color (H) from intensity (V). Easy to isolate specific colors.
- **Con:** Hue undefined for grayscale.
- **Use:** Color-based segmentation, tracking.

**LAB:**
- **Pro:** Perceptually uniform (equal distances = equal perceived differences).
- **Con:** Not intuitive, requires conversion.
- **Use:** Color matching, image enhancement.

## Q3: What is the Nyquist-Shannon sampling theorem for images?
**Answer:**
To avoid aliasing, the **sampling rate must be at least twice the highest spatial frequency** in the image:
$$ f_s \geq 2 f_{max} $$

**In images:** If you downsample without low-pass filtering first, high-frequency details (fine edges, textures) will create artifacts (moiré patterns, jaggies).

**Solution:** Apply Gaussian blur before downsampling to remove high frequencies.

## Q4: What is aliasing and how does it manifest in images?
**Answer:**
**Aliasing** occurs when high-frequency information is sampled too sparsely, causing it to appear as false low-frequency patterns.

**Examples:**
- **Spatial aliasing:** Jagged edges on diagonals (staircase effect).
- **Moiré patterns:** Regular textures (bricks, fabric) create interference patterns.
- **Wagon wheel effect (temporal):** Wheels appear to spin backward in videos.

**Prevention:**
- **Anti-aliasing:** Blur before downsampling (low-pass filter).
- **Super-sampling:** Sample at higher resolution, then downsample.

## Q5: Explain the pinhole camera model.
**Answer:**
The **pinhole camera model** describes perspective projection from 3D world to 2D image:

$$ \begin{bmatrix} u \\ v \\ 1 \end{bmatrix} = \frac{f}{Z} \begin{bmatrix} X \\ Y \\ 1 \end{bmatrix} $$

- **$(X, Y, Z)$:** 3D point in camera coordinates.
- **$(u, v)$:** 2D image coordinates.
- **$f$:** Focal length.

**Key properties:**
- Straight lines remain straight.
- Parallel lines converge to vanishing points.
- Objects farther away appear smaller (perspective).

## Q6: What is dynamic range and why is it important?
**Answer:**
**Dynamic range** is the ratio between the brightest and darkest values the sensor can capture:
$$ DR = 20 \log_{10}(I_{max}/I_{min}) \text{ dB} $$

**Importance:**
- **Low DR (< 60 dB):** Bright areas blown out OR dark areas too noisy (can't have both).
- **High DR (> 100 dB):** Capture detail in both shadows and highlights simultaneously.

**Human eye:** ~100-150 dB dynamic range. Standard cameras: ~50-70 dB.

**HDR imaging:** Combine multiple exposures to extend dynamic range.

## Q7: What is the difference between image resolution and image size?
**Answer:**
- **Image size:** Physical dimensions in pixels (e.g., 1920×1080 pixels).
- **Image resolution:** Pixel density per physical unit (e.g., 300 DPI - dots per inch).

**Example:**
- 1920×1080 image at 100 DPI → 19.2" × 10.8" print.
- Same image at 300 DPI → 6.4" × 3.6" print.

Higher resolution = more detail captured per physical area.

## Q8: How do you efficiently manipulate images in Python?
**Answer:**
Use **NumPy** for vectorized operations:

```python
import numpy as np

# Load image (H, W, 3)
img = np.array(Image.open('img.jpg'))

# Normalize to [0, 1]
img_norm = img.astype(np.float32) / 255.0

# Grayscale conversion
gray = 0.299*img[:,:,0] + 0.587*img[:,:,1] + 0.114*img[:,:,2]

# Flip, crop, rotate
flipped = np.flip(img, axis=1)
cropped = img[100:300, 200:400]
rotated = np.rot90(img)

# Channel statistics
mean_rgb = np.mean(img, axis=(0,1))  # Per-channel mean
```

**Why NumPy:** Vectorized operations are 10-100x faster than Python loops.

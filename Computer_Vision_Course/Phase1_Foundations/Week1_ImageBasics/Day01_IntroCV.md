# Day 1: Introduction to Computer Vision & Image Fundamentals

## 1. What is Computer Vision?
Computer Vision is the field of artificial intelligence that trains computers to **interpret and understand visual information** from the world, mimicking human visual perception.

**Core Tasks:**
- **Classification:** What is in the image?
- **Detection:** Where are objects located?
- **Segmentation:** What is the precise shape of each object?
- **Generation:** Create new visual content.
- **Understanding:** Describe scenes, relationships, actions.

## 2. Digital Images: Mathematical Representation
An image is a function $I: \mathbb{R}^2 \rightarrow \mathbb{R}^n$ mapping spatial coordinates to intensity values.

### Grayscale Images
$$ I(x, y) \in [0, 255] $$
where $(x, y)$ are pixel coordinates and the value represents intensity.

**Discrete representation:** An $M \times N$ matrix where $I[i, j]$ is the intensity at pixel $(i, j)$.

### Color Images (RGB)
$$ I(x, y) = \begin{bmatrix} R(x,y) \\ G(x,y) \\ B(x,y) \end{bmatrix} $$
Each channel is an $M \times N$ matrix, resulting in an $M \times N \times 3$ tensor.

## 3. Color Spaces
Different representations of color information:

### RGB (Red, Green, Blue)
- **Additive color model:** Mix light.
- **Range:** Each channel $\in [0, 255]$ (8-bit).
- **Use:** Display devices, cameras.

### HSV (Hue, Saturation, Value)
- **Hue:** Color type (0-360°).
- **Saturation:** Color purity (0-100%).
- **Value:** Brightness (0-100%).
- **Use:** Color-based segmentation, easier to isolate colors.

### LAB (Lightness, A, B)
- **L:** Lightness (0-100).
- **A:** Green-Red axis.
- **B:** Blue-Yellow axis.
- **Perceptually uniform:** Equal color distances look equally different.

### Conversion Example: RGB to Grayscale
$$ I_{gray}(x, y) = 0.299 \cdot R + 0.587 \cdot G + 0.114 \cdot B $$
Weights account for human eye sensitivity (more sensitive to green).

## 4. Image Properties & Challenges

### Illumination
Light changes affect pixel values:
$$ I_{observed}(x, y) = \rho(x, y) \cdot L(x, y) $$
where $\rho$ is surface reflectance and $L$ is illumination.

### Noise
Images contain random variations:
- **Gaussian noise:** $I_{noisy} = I + \mathcal{N}(0, \sigma^2)$
- **Salt-and-pepper:** Random pixels set to min/max.
- **Poisson noise:** Photon counting (low light).

### Scale & Resolution
- **Spatial resolution:** Pixel density (e.g., 1920×1080).
- **Color depth:** Bits per pixel (8-bit, 16-bit, 32-bit float).

## 5. Basic Image Operations in NumPy
```python
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load image
img = Image.open('image.jpg')
img_array = np.array(img)  # Shape: (H, W, 3) for RGB

# Access pixels
pixel_value = img_array[100, 150]  # RGB at (100, 150)

# Convert to grayscale (manual)
gray = 0.299*img_array[:,:,0] + 0.587*img_array[:,:,1] + 0.114*img_array[:,:,2]

# Crop image
cropped = img_array[100:300, 200:400]  # [y1:y2, x1:x2]

# Flip
flipped_horizontal = np.flip(img_array, axis=1)
flipped_vertical = np.flip(img_array, axis=0)

# Rotate 90 degrees
rotated = np.rot90(img_array)

# Normalize to [0, 1]
normalized = img_array.astype(np.float32) / 255.0

# Channel statistics
mean_per_channel = np.mean(img_array, axis=(0, 1))  # (R_mean, G_mean, B_mean)
std_per_channel = np.std(img_array, axis=(0, 1))
```

## 6. Image Descriptors & Statistics

### Histogram
Frequency distribution of pixel intensities:
$$ h(i) = \sum_{x,y} \mathbf{1}[I(x,y) = i] $$

### Moments
- **Mean:** $\bar{I} = \frac{1}{MN} \sum_{x,y} I(x,y)$
- **Variance:** $\sigma^2 = \frac{1}{MN} \sum_{x,y} (I(x,y) - \bar{I})^2$
- **Entropy:** $H = -\sum_i p(i) \log p(i)$ (measures randomness)

## 7. Applications of Computer Vision
- **Medical Imaging:** X-ray analysis, tumor detection, cell counting.
- **Autonomous Vehicles:** Lane detection, pedestrian detection, scene understanding.
- **Face Recognition:** Security, authentication, photo organization.
- **Robotics:** Visual servoing, object manipulation, navigation.
- **AR/VR:** Pose estimation, scene reconstruction, tracking.
- **Agriculture:** Crop monitoring, disease detection, yield prediction.
- **Retail:** Product recognition, inventory management, checkout automation.

### Key Takeaways
- Images are discrete samples of continuous visual information.
- Color spaces offer different representations for different tasks.
- Understanding image properties is crucial for algorithm design.
- NumPy is the foundation for efficient image manipulation in Python.

# Lab 05: Gaussian Blur Implementation from Scratch

## Difficulty
🟡 **Medium**

## Estimated Time
1 hour

## Learning Objectives
- Understand convolution operations in image processing
- Implement Gaussian kernel generation
- Apply spatial filtering techniques
- Optimize convolution for performance
- Compare with library implementations

## Prerequisites
- Day 2: Image Processing Operations
- Understanding of NumPy arrays
- Basic knowledge of convolution
- Familiarity with image representations

## Problem Statement

Implement Gaussian blur from scratch without using OpenCV's `GaussianBlur` function. Your implementation should:

1. Generate a Gaussian kernel of specified size and sigma
2. Apply the kernel to an image using convolution
3. Handle edge cases (image boundaries)
4. Support both grayscale and color images
5. Match the quality of OpenCV's implementation

### What is Gaussian Blur?

Gaussian blur is a widely-used image smoothing technique that reduces noise and detail. It works by convolving the image with a Gaussian kernel, which gives more weight to nearby pixels and less weight to distant pixels.

**Gaussian Function (2D)**:
```
G(x, y) = (1 / (2π σ²)) * e^(-(x² + y²) / (2σ²))
```

Where:
- `σ` (sigma) controls the amount of blur
- Larger σ → more blur
- Smaller σ → less blur

### Example

```python
Input Image: 
[[100, 120, 110],
 [115, 125, 120],
 [105, 115, 110]]

Gaussian Kernel (3x3, σ=1.0):
[[0.075, 0.124, 0.075],
 [0.124, 0.204, 0.124],
 [0.075, 0.124, 0.075]]

Output (blurred image):
[[114.5, 117.8, 115.2],
 [113.2, 116.5, 114.8],
 [111.8, 114.2, 112.5]]
```

## Requirements

1. Implement `generate_gaussian_kernel(kernel_size, sigma)` function
2. Implement `gaussian_blur(image, kernel_size, sigma)` function
3. Handle edge padding (reflect, constant, or wrap)
4. Support both grayscale (H×W) and color (H×W×3) images
5. Validate kernel normalization (sum = 1)
6. Compare results with OpenCV's implementation

## Starter Code

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

def generate_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    Args:
        kernel_size: Size of the kernel (must be odd)
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        2D numpy array representing the Gaussian kernel
    """
    # TODO: Implement Gaussian kernel generation
    pass

def gaussian_blur(image: np.ndarray, kernel_size: int = 5, sigma: float = 1.0) -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image (grayscale or RGB)
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
        
    Returns:
        Blurred image
    """
    # TODO: Implement Gaussian blur
    pass

# Test code
def test_gaussian_blur():
    # Create a simple test image
    test_image = np.array([
        [100, 120, 110, 100],
        [115, 125, 120, 105],
        [105, 115, 110, 100],
        [100, 110, 105, 95]
    ], dtype=np.float32)
    
    # Apply blur
    blurred = gaussian_blur(test_image, kernel_size=3, sigma=1.0)
    
    # Verify output shape
    assert blurred.shape == test_image.shape
    
    # Verify blurring effect (center should be smoothed)
    assert abs(blurred[1, 1] - 115) < 5  # Should be close to average
    
    print("✅ Basic tests passed!")

if __name__ == "__main__":
    test_gaussian_blur()
```

## Hints

<details>
<summary>Hint 1: Generating Gaussian Kernel</summary>

Create a grid of (x, y) coordinates centered at (0, 0):
```python
ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
xx, yy = np.meshgrid(ax, ax)
```

Then apply the Gaussian formula:
```python
kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
kernel = kernel / np.sum(kernel)  # Normalize
```
</details>

<details>
<summary>Hint 2: Convolution Implementation</summary>

For each pixel, multiply the kernel with the surrounding pixels and sum:
```python
for i in range(height):
    for j in range(width):
        region = padded_image[i:i+k_size, j:j+k_size]
        output[i, j] = np.sum(region * kernel)
```
</details>

<details>
<summary>Hint 3: Handling Edges</summary>

Pad the image before convolution to handle boundaries:
```python
pad_size = kernel_size // 2
padded = np.pad(image, pad_size, mode='reflect')
```

Modes: 'reflect', 'constant', 'edge', 'wrap'
</details>

<details>
<summary>Hint 4: Color Images</summary>

For RGB images, apply blur to each channel separately:
```python
if len(image.shape) == 3:
    return np.stack([gaussian_blur(image[:,:,c]) for c in range(3)], axis=2)
```
</details>

## Solution

<details>
<summary>Click to reveal solution</summary>

### Complete Implementation

```python
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Optional

def generate_gaussian_kernel(kernel_size: int, sigma: float) -> np.ndarray:
    """
    Generate a 2D Gaussian kernel.
    
    The Gaussian function in 2D is:
    G(x, y) = (1 / (2π σ²)) * exp(-(x² + y²) / (2σ²))
    
    Args:
        kernel_size: Size of the kernel (must be odd, e.g., 3, 5, 7)
        sigma: Standard deviation of the Gaussian distribution
        
    Returns:
        Normalized 2D Gaussian kernel
    """
    # Validate inputs
    if kernel_size % 2 == 0:
        raise ValueError("Kernel size must be odd")
    
    if sigma <= 0:
        raise ValueError("Sigma must be positive")
    
    # Create coordinate grids
    # For kernel_size=5: ax = [-2, -1, 0, 1, 2]
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(ax, ax)
    
    # Apply Gaussian formula
    # G(x,y) = exp(-(x² + y²) / (2σ²))
    # We omit the constant factor (1 / 2πσ²) since we normalize anyway
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    
    # Normalize so sum = 1 (preserves image brightness)
    kernel = kernel / np.sum(kernel)
    
    return kernel

def convolve2d(image: np.ndarray, kernel: np.ndarray, 
               padding: str = 'reflect') -> np.ndarray:
    """
    Perform 2D convolution on an image with a kernel.
    
    Args:
        image: 2D input image
        kernel: 2D convolution kernel
        padding: Padding mode ('reflect', 'constant', 'edge', 'wrap')
        
    Returns:
        Convolved image
    """
    # Get dimensions
    img_height, img_width = image.shape
    kernel_size = kernel.shape[0]
    pad_size = kernel_size // 2
    
    # Pad the image
    if padding == 'constant':
        padded = np.pad(image, pad_size, mode='constant', constant_values=0)
    else:
        padded = np.pad(image, pad_size, mode=padding)
    
    # Initialize output
    output = np.zeros_like(image)
    
    # Perform convolution
    for i in range(img_height):
        for j in range(img_width):
            # Extract region
            region = padded[i:i+kernel_size, j:j+kernel_size]
            
            # Element-wise multiplication and sum
            output[i, j] = np.sum(region * kernel)
    
    return output

def gaussian_blur(image: np.ndarray, kernel_size: int = 5, 
                  sigma: float = 1.0, padding: str = 'reflect') -> np.ndarray:
    """
    Apply Gaussian blur to an image.
    
    Args:
        image: Input image (grayscale or RGB)
        kernel_size: Size of the Gaussian kernel (must be odd)
        sigma: Standard deviation for Gaussian kernel
        padding: Padding mode for edges
        
    Returns:
        Blurred image with same shape as input
    """
    # Generate Gaussian kernel
    kernel = generate_gaussian_kernel(kernel_size, sigma)
    
    # Handle grayscale vs color images
    if len(image.shape) == 2:
        # Grayscale image
        return convolve2d(image, kernel, padding)
    
    elif len(image.shape) == 3:
        # Color image (RGB)
        channels = image.shape[2]
        output = np.zeros_like(image)
        
        # Apply blur to each channel independently
        for c in range(channels):
            output[:, :, c] = convolve2d(image[:, :, c], kernel, padding)
        
        return output
    
    else:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")

# Optimized version using NumPy's advanced indexing
def gaussian_blur_optimized(image: np.ndarray, kernel_size: int = 5, 
                           sigma: float = 1.0) -> np.ndarray:
    """
    Optimized Gaussian blur using separable filters.
    
    Gaussian kernels are separable: 2D kernel = 1D_x * 1D_y^T
    This reduces complexity from O(k²) to O(2k) per pixel.
    """
    # Generate 1D Gaussian kernel
    ax = np.arange(-kernel_size // 2 + 1, kernel_size // 2 + 1)
    kernel_1d = np.exp(-(ax**2) / (2 * sigma**2))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    
    # Pad image
    pad_size = kernel_size // 2
    padded = np.pad(image, pad_size, mode='reflect')
    
    # Apply horizontal convolution
    temp = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            temp[i, j] = np.sum(padded[i+pad_size, j:j+kernel_size] * kernel_1d)
    
    # Pad temp
    padded_temp = np.pad(temp, pad_size, mode='reflect')
    
    # Apply vertical convolution
    output = np.zeros_like(image)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            output[i, j] = np.sum(padded_temp[i:i+kernel_size, j+pad_size] * kernel_1d)
    
    return output
```

### Detailed Walkthrough

Let's trace through a simple example:

**Input Image (3×3)**:
```
[[100, 120, 110],
 [115, 125, 120],
 [105, 115, 110]]
```

**Step 1: Generate Gaussian Kernel (3×3, σ=1.0)**
```python
ax = [-1, 0, 1]
xx, yy = meshgrid(ax, ax)

# xx:          yy:
# [[-1, 0, 1], [[-1, -1, -1],
#  [-1, 0, 1],  [ 0,  0,  0],
#  [-1, 0, 1]]  [ 1,  1,  1]]

# Distance squared from center:
# xx² + yy² = [[2, 1, 2],
#              [1, 0, 1],
#              [2, 1, 2]]

# Gaussian values (before normalization):
# exp(-dist² / 2σ²) = [[0.1353, 0.6065, 0.1353],
#                      [0.6065, 1.0000, 0.6065],
#                      [0.1353, 0.6065, 0.1353]]

# After normalization (sum = 1):
kernel = [[0.0751, 0.1238, 0.0751],
          [0.1238, 0.2042, 0.1238],
          [0.0751, 0.1238, 0.0751]]
```

**Step 2: Pad Image (reflect mode)**
```
Padded (5×5):
[[125, 115, 125, 120, 110],
 [120, 100, 120, 110, 100],
 [125, 115, 125, 120, 110],
 [115, 105, 115, 110, 100],
 [125, 115, 125, 120, 110]]
```

**Step 3: Convolve Center Pixel (1, 1)**
```
Region around (1,1):
[[125, 115, 125],
 [120, 100, 120],
 [125, 115, 125]]

Convolution:
= 125*0.0751 + 115*0.1238 + 125*0.0751
+ 120*0.1238 + 100*0.2042 + 120*0.1238
+ 125*0.0751 + 115*0.1238 + 125*0.0751

= 9.39 + 14.24 + 9.39
+ 14.86 + 20.42 + 14.86
+ 9.39 + 14.24 + 9.39

= 116.18

Output[1,1] ≈ 116
```

### Comparison with OpenCV

```python
import cv2

def compare_with_opencv():
    # Load test image
    image = cv2.imread('test.jpg', cv2.IMREAD_GRAYSCALE).astype(np.float32)
    
    # Our implementation
    our_blur = gaussian_blur(image, kernel_size=5, sigma=1.5)
    
    # OpenCV implementation
    cv_blur = cv2.GaussianBlur(image, (5, 5), 1.5)
    
    # Calculate difference
    diff = np.abs(our_blur - cv_blur)
    max_diff = np.max(diff)
    mean_diff = np.mean(diff)
    
    print(f"Max difference: {max_diff:.4f}")
    print(f"Mean difference: {mean_diff:.4f}")
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(our_blur, cmap='gray')
    axes[0].set_title('Our Implementation')
    axes[1].imshow(cv_blur, cmap='gray')
    axes[1].set_title('OpenCV')
    axes[2].imshow(diff, cmap='hot')
    axes[2].set_title(f'Difference (max: {max_diff:.2f})')
    plt.show()
```

</details>

## Test Cases

```python
def comprehensive_tests():
    """Comprehensive test suite."""
    
    # Test 1: Kernel generation
    kernel = generate_gaussian_kernel(3, 1.0)
    assert kernel.shape == (3, 3)
    assert np.isclose(np.sum(kernel), 1.0)  # Normalized
    assert kernel[1, 1] > kernel[0, 0]  # Center has highest value
    
    # Test 2: Grayscale image
    gray_img = np.random.rand(100, 100) * 255
    blurred = gaussian_blur(gray_img, 5, 1.0)
    assert blurred.shape == gray_img.shape
    
    # Test 3: Color image
    color_img = np.random.rand(100, 100, 3) * 255
    blurred_color = gaussian_blur(color_img, 5, 1.0)
    assert blurred_color.shape == color_img.shape
    
    # Test 4: Edge preservation
    # Create image with sharp edge
    edge_img = np.zeros((50, 50))
    edge_img[:, 25:] = 255
    blurred_edge = gaussian_blur(edge_img, 5, 1.0)
    # Edge should be smoothed but not completely gone
    assert blurred_edge[25, 24] > 0 and blurred_edge[25, 24] < 255
    
    # Test 5: Different sigmas
    img = np.random.rand(50, 50) * 255
    blur_small = gaussian_blur(img, 5, 0.5)
    blur_large = gaussian_blur(img, 5, 2.0)
    # Larger sigma should blur more
    variance_small = np.var(blur_small)
    variance_large = np.var(blur_large)
    assert variance_small > variance_large
    
    print("✅ All comprehensive tests passed!")
```

## Extensions

1. **Bilateral Filter**: Implement bilateral filtering that preserves edges while blurring
2. **Anisotropic Gaussian**: Create elongated Gaussian kernels for directional blur
3. **GPU Acceleration**: Implement using CuPy or PyTorch for GPU speedup
4. **Adaptive Sigma**: Vary sigma based on local image content
5. **Real-time Video**: Apply Gaussian blur to video streams efficiently

## Related Concepts
- [Day 2: Image Processing Operations](../Day02_ImageProcessing.md)
- [Lab 06: Median Filter](lab_06_median_filter.md)
- [Lab 07: Bilateral Filter](lab_07_bilateral_filter.md)

## Real-World Applications

1. **Noise Reduction**: Preprocessing for object detection
2. **Image Pyramids**: Multi-scale image analysis
3. **Edge Detection**: Preprocessing for Canny edge detector
4. **Background Blur**: Portrait mode in smartphones
5. **Medical Imaging**: Denoising X-rays and MRI scans

## Performance Analysis

```python
import time

def benchmark():
    sizes = [100, 200, 500, 1000]
    kernel_size = 5
    sigma = 1.5
    
    for size in sizes:
        img = np.random.rand(size, size) * 255
        
        start = time.time()
        _ = gaussian_blur(img, kernel_size, sigma)
        elapsed = time.time() - start
        
        print(f"Image size {size}×{size}: {elapsed:.3f}s")
```

**Expected Complexity**:
- Time: O(n × m × k²) where n×m is image size, k is kernel size
- Space: O(n × m) for padded image
- Optimized (separable): O(n × m × 2k)

## Key Takeaways

1. **Gaussian Kernel**: Weights decrease with distance from center
2. **Normalization**: Kernel must sum to 1 to preserve brightness
3. **Separability**: 2D Gaussian = 1D horizontal × 1D vertical (optimization)
4. **Padding**: Essential for handling image boundaries
5. **Sigma Parameter**: Controls blur strength (larger σ = more blur)

---

**Next**: [Lab 06: Median Filter Implementation](lab_06_median_filter.md)

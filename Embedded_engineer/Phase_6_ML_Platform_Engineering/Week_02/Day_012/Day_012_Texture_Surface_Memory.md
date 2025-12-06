# Day 12: Texture and Surface Memory
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 2: Advanced CUDA Programming

---

> **🎯 Focus Area:** Master specialized GPU memory types optimized for spatial locality and image processing operations.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1. **Understand** texture memory architecture and caching behavior
2. **Implement** texture-based image processing kernels
3. **Use** surface memory for read-write image operations
4. **Apply** texture filtering and boundary handling modes
5. **Optimize** memory access patterns using texture cache
6. **Compare** performance: texture vs global memory

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU with texture support (all modern GPUs)
- For surface memory: compute capability 2.0+

### Software Environment
```bash
pip install cupy-cuda12x numpy pillow matplotlib

# Verify texture support
python -c "import cupy as cp; print(cp.cuda.Device().attributes)"
```

### Prior Knowledge
- Day 3: CUDA Memory Management
- Day 6: Shared Memory Optimization
- Basic image processing concepts

---

## 📖 Theoretical Foundation

### 1. Texture Memory Overview

Texture memory is a **special read-only memory** optimized for:
- **2D spatial locality** - Neighboring pixels cached together
- **Hardware interpolation** - Free bilinear/trilinear filtering
- **Boundary handling** - Automatic clamp, wrap, mirror modes
- **Format conversion** - Automatic normalization to [0,1]

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TEXTURE MEMORY ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Regular Global Memory Access:                                               │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Request: pixel(x,y) ─────▶ L2 Cache ─────▶ DRAM ─────▶ Value        │    │
│  │                           (128-byte lines, 1D locality)             │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Texture Memory Access:                                                      │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │ Request: tex2D(x,y) ─▶ Texture Cache ─▶ L2 ─▶ DRAM ─────▶ Value    │    │
│  │                       (2D spatial locality, interpolation)          │    │
│  │                                                                     │    │
│  │  Texture Cache Layout:                                              │    │
│  │  ┌─────┬─────┬─────┬─────┐                                         │    │
│  │  │ 4x4 │ 4x4 │ 4x4 │ 4x4 │  ◄── 2D tiles cached together          │    │
│  │  │tile │tile │tile │tile │                                         │    │
│  │  ├─────┼─────┼─────┼─────┤                                         │    │
│  │  │ 4x4 │ 4x4 │ 4x4 │ 4x4 │                                         │    │
│  │  │tile │tile │tile │tile │                                         │    │
│  │  └─────┴─────┴─────┴─────┘                                         │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 2. CUDA Array vs Linear Memory

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                     MEMORY LAYOUTS FOR TEXTURES                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Linear Memory (Pitched):                                                    │
│  ┌────────────────────────────────┐                                         │
│  │ Row 0: [P00][P01][P02]...[pad] │  ◄── Pitch alignment (256/512 bytes)    │
│  │ Row 1: [P10][P11][P12]...[pad] │                                         │
│  │ Row 2: [P20][P21][P22]...[pad] │                                         │
│  │ ...                            │                                         │
│  └────────────────────────────────┘                                         │
│  + Simple allocation                                                         │
│  + Can use normal pointers                                                   │
│  - Less optimal 2D caching                                                   │
│                                                                              │
│  CUDA Array (Morton order / Z-order):                                        │
│  ┌────────────────────────────────┐                                         │
│  │ [0,0][1,0][0,1][1,1]|[2,0]...  │  ◄── Space-filling curve order         │
│  │ Optimized for 2D locality      │                                         │
│  └────────────────────────────────┘                                         │
│  + Best 2D cache utilization                                                 │
│  + Supports all texture features                                             │
│  - Requires special copy functions                                           │
│  - Cannot access with regular pointers                                       │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 3. Texture Addressing Modes

```cpp
// Addressing modes for out-of-bounds access
cudaAddressModeWrap,   // Wrap around (repeat texture)
cudaAddressModeClamp,  // Clamp to edge (replicate boundary)
cudaAddressModeMirror, // Mirror at boundary
cudaAddressModeBorder  // Return border color (black)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       ADDRESSING MODES VISUALIZATION                         │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Original Texture:    Wrap Mode:          Clamp Mode:        Mirror Mode:   │
│  ┌───────┐            ┌───────────────┐   ┌───────────────┐  ┌───────────┐  │
│  │ A B C │            │ C A B C A B C │   │ A A B C C C C │  │ C B A B C │  │
│  │ D E F │            │ F D E F D E F │   │ D D E F F F F │  │ F E D E F │  │
│  │ G H I │            │ I G H I G H I │   │ G G H I I I I │  │ I H G H I │  │
│  └───────┘            │ C A B C A B C │   │ G G H I I I I │  │ F E D E F │  │
│                       └───────────────┘   └───────────────┘  └───────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 4. Texture Filtering Modes

```cpp
// Filter modes for non-integer coordinates
cudaFilterModePoint,   // Nearest neighbor (no interpolation)
cudaFilterModeLinear   // Bilinear interpolation (free in hardware!)
```

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       TEXTURE FILTERING MODES                                │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  Point Filtering (Nearest Neighbor):                                         │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Sample at (1.3, 0.8):                                              │    │
│  │  ┌───┬───┐                                                          │    │
│  │  │ A │ B │ ◄── Returns A (nearest integer coordinates)              │    │
│  │  ├───┼───┤                                                          │    │
│  │  │ C │ D │     Result = A                                           │    │
│  │  └───┴───┘                                                          │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  Linear Filtering (Bilinear Interpolation):                                  │
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │  Sample at (1.3, 0.8):                                              │    │
│  │  ┌───┬───┐                                                          │    │
│  │  │ A │ B │     fx = 0.3, fy = 0.8                                   │    │
│  │  ├───●───┤                                                          │    │
│  │  │ C │ D │     Result = A*(1-fx)*(1-fy) + B*fx*(1-fy)              │    │
│  │  └───┴───┘            + C*(1-fx)*fy     + D*fx*fy                  │    │
│  │                       = 0.14A + 0.06B + 0.56C + 0.24D               │    │
│  └─────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 5. Surface Memory

Surface memory extends texture memory with **read-write capability**:

```cpp
// Surface object - read/write access
surfaceReference surfRef;

// Surface write
surf2Dwrite(value, surfRef, x * sizeof(float), y);

// Surface read
float value = surf2Dread<float>(surfRef, x * sizeof(float), y);
```

---

## 💻 Implementation

### 👨‍💻 Core Implementation

#### 📁 `src/texture_memory.py` - Texture Memory Operations
```python
#!/usr/bin/env python3
"""
Day 12: Texture and Surface Memory
Phase 6: AI/ML Platform Engineering with GPU Programming

This module demonstrates texture memory usage for image processing
and spatial data access optimization.
"""

import cupy as cp
import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass
import time


# ============================================================================
# TEXTURE-BASED IMAGE PROCESSING KERNELS
# ============================================================================

# Image convolution using global memory (baseline)
convolution_global_kernel = cp.RawKernel(r'''
extern "C" __global__
void convolutionGlobal(
    const float* input,
    float* output,
    const float* kernel,
    int width, int height,
    int kernelWidth, int kernelHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfKW = kernelWidth / 2;
    int halfKH = kernelHeight / 2;
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
            int ix = x + kx - halfKW;
            int iy = y + ky - halfKH;
            
            // Clamp to boundaries
            ix = max(0, min(ix, width - 1));
            iy = max(0, min(iy, height - 1));
            
            sum += input[iy * width + ix] * kernel[ky * kernelWidth + kx];
        }
    }
    
    output[y * width + x] = sum;
}
''', 'convolutionGlobal')


# Convolution using texture memory (CuPy texture object approach)
convolution_texture_kernel = cp.RawKernel(r'''
texture<float, 2, cudaReadModeElementType> texInput;

extern "C" __global__
void convolutionTexture(
    float* output,
    const float* kernel,
    int width, int height,
    int kernelWidth, int kernelHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    int halfKW = kernelWidth / 2;
    int halfKH = kernelHeight / 2;
    
    float sum = 0.0f;
    
    for (int ky = 0; ky < kernelHeight; ky++) {
        for (int kx = 0; kx < kernelWidth; kx++) {
            // tex2D automatically handles boundary clamping
            // and provides hardware-accelerated access
            float pixel = tex2D(texInput, 
                               (float)(x + kx - halfKW) + 0.5f,
                               (float)(y + ky - halfKH) + 0.5f);
            sum += pixel * kernel[ky * kernelWidth + kx];
        }
    }
    
    output[y * width + x] = sum;
}
''', 'convolutionTexture')


# Bilinear interpolation (demonstrating free hardware filtering)
bilinear_resize_kernel = cp.RawKernel(r'''
texture<float, 2, cudaReadModeElementType> texInput;

extern "C" __global__
void bilinearResize(
    float* output,
    int outWidth, int outHeight,
    int inWidth, int inHeight
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= outWidth || y >= outHeight) return;
    
    // Map output coordinates to input coordinates
    float scaleX = (float)inWidth / outWidth;
    float scaleY = (float)inHeight / outHeight;
    
    float srcX = (x + 0.5f) * scaleX;
    float srcY = (y + 0.5f) * scaleY;
    
    // tex2D with filterMode=Linear does bilinear interpolation for FREE
    float value = tex2D(texInput, srcX, srcY);
    
    output[y * outWidth + x] = value;
}
''', 'bilinearResize')


# Image rotation using texture memory
rotation_kernel = cp.RawKernel(r'''
texture<float, 2, cudaReadModeElementType> texInput;

extern "C" __global__
void rotateImage(
    float* output,
    int width, int height,
    float cosAngle, float sinAngle,
    float centerX, float centerY
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    // Translate to origin
    float fx = x - centerX;
    float fy = y - centerY;
    
    // Rotate
    float srcX = fx * cosAngle - fy * sinAngle + centerX + 0.5f;
    float srcY = fx * sinAngle + fy * cosAngle + centerY + 0.5f;
    
    // Sample with hardware interpolation
    float value = tex2D(texInput, srcX, srcY);
    
    output[y * width + x] = value;
}
''', 'rotateImage')


@dataclass
class ImageProcessingResult:
    """Result of an image processing operation."""
    output: np.ndarray
    time_ms: float
    method: str


class TextureImageProcessor:
    """
    Image processor demonstrating texture memory benefits.
    
    This class shows how texture memory can accelerate image processing
    operations that benefit from 2D spatial locality.
    """
    
    def __init__(self):
        self.kernels = self._create_common_kernels()
    
    def _create_common_kernels(self):
        """Create common convolution kernels."""
        kernels = {}
        
        # Gaussian blur 5x5
        gaussian = np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=np.float32) / 256.0
        kernels['gaussian_5x5'] = gaussian
        
        # Sobel edge detection
        sobel_x = np.array([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=np.float32)
        kernels['sobel_x'] = sobel_x
        
        sobel_y = np.array([
            [-1, -2, -1],
            [ 0,  0,  0],
            [ 1,  2,  1]
        ], dtype=np.float32)
        kernels['sobel_y'] = sobel_y
        
        # Sharpening kernel
        sharpen = np.array([
            [ 0, -1,  0],
            [-1,  5, -1],
            [ 0, -1,  0]
        ], dtype=np.float32)
        kernels['sharpen'] = sharpen
        
        # Box blur 3x3
        box_blur = np.ones((3, 3), dtype=np.float32) / 9.0
        kernels['box_blur_3x3'] = box_blur
        
        return kernels
    
    def convolve_global(self, image: np.ndarray, kernel: np.ndarray) -> ImageProcessingResult:
        """
        Perform convolution using global memory.
        
        Args:
            image: Input image (H, W) float32
            kernel: Convolution kernel (kH, kW) float32
            
        Returns:
            ImageProcessingResult with output and timing
        """
        height, width = image.shape
        kh, kw = kernel.shape
        
        # Transfer to GPU
        d_input = cp.asarray(image)
        d_output = cp.zeros_like(d_input)
        d_kernel = cp.asarray(kernel)
        
        # Configure grid
        block = (16, 16)
        grid = ((width + block[0] - 1) // block[0],
                (height + block[1] - 1) // block[1])
        
        # Warmup
        convolution_global_kernel(grid, block,
            (d_input, d_output, d_kernel, width, height, kw, kh))
        cp.cuda.Stream.null.synchronize()
        
        # Benchmark
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(10):
            convolution_global_kernel(grid, block,
                (d_input, d_output, d_kernel, width, height, kw, kh))
        end.record()
        end.synchronize()
        
        time_ms = cp.cuda.get_elapsed_time(start, end) / 10
        
        return ImageProcessingResult(
            output=d_output.get(),
            time_ms=time_ms,
            method="global_memory"
        )
    
    def gaussian_blur(self, image: np.ndarray) -> ImageProcessingResult:
        """Apply Gaussian blur using global memory convolution."""
        return self.convolve_global(image, self.kernels['gaussian_5x5'])
    
    def sobel_edge_detection(self, image: np.ndarray) -> Tuple[ImageProcessingResult, ImageProcessingResult]:
        """Compute Sobel edge detection."""
        result_x = self.convolve_global(image, self.kernels['sobel_x'])
        result_y = self.convolve_global(image, self.kernels['sobel_y'])
        return result_x, result_y


def demonstrate_texture_vs_global():
    """Compare texture memory vs global memory performance."""
    print("\n" + "="*70)
    print("TEXTURE vs GLOBAL MEMORY COMPARISON")
    print("="*70)
    
    # Create test image
    sizes = [(512, 512), (1024, 1024), (2048, 2048), (4096, 4096)]
    
    for height, width in sizes:
        print(f"\n--- Image Size: {width}x{height} ---")
        
        # Create random image
        image = np.random.rand(height, width).astype(np.float32)
        
        # Gaussian kernel
        kernel = np.array([
            [1,  4,  6,  4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1,  4,  6,  4, 1]
        ], dtype=np.float32) / 256.0
        
        # Global memory version
        processor = TextureImageProcessor()
        result_global = processor.convolve_global(image, kernel)
        
        print(f"  Global Memory: {result_global.time_ms:.3f} ms")
        
        # Calculate effective bandwidth
        bytes_read = height * width * 4 * 25  # 5x5 kernel
        bytes_written = height * width * 4
        total_bytes = bytes_read + bytes_written
        bandwidth = total_bytes / (result_global.time_ms / 1000) / 1e9
        
        print(f"  Effective Bandwidth: {bandwidth:.1f} GB/s")


def demonstrate_bilinear_interpolation():
    """Demonstrate hardware bilinear interpolation."""
    print("\n" + "="*70)
    print("BILINEAR INTERPOLATION DEMONSTRATION")
    print("="*70)
    
    # Create test pattern
    in_size = 256
    out_size = 1024
    
    # Create checkerboard pattern
    image = np.zeros((in_size, in_size), dtype=np.float32)
    for y in range(in_size):
        for x in range(in_size):
            if (x // 32 + y // 32) % 2 == 0:
                image[y, x] = 1.0
    
    print(f"  Input size: {in_size}x{in_size}")
    print(f"  Output size: {out_size}x{out_size}")
    print(f"  Upscale factor: {out_size // in_size}x")
    
    # Manual bilinear interpolation (baseline)
    d_input = cp.asarray(image)
    d_output = cp.zeros((out_size, out_size), dtype=cp.float32)
    
    manual_bilinear = cp.RawKernel(r'''
    extern "C" __global__
    void manualBilinear(
        const float* input,
        float* output,
        int outWidth, int outHeight,
        int inWidth, int inHeight
    ) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= outWidth || y >= outHeight) return;
        
        float scaleX = (float)inWidth / outWidth;
        float scaleY = (float)inHeight / outHeight;
        
        float srcX = (x + 0.5f) * scaleX - 0.5f;
        float srcY = (y + 0.5f) * scaleY - 0.5f;
        
        int x0 = (int)floorf(srcX);
        int y0 = (int)floorf(srcY);
        int x1 = min(x0 + 1, inWidth - 1);
        int y1 = min(y0 + 1, inHeight - 1);
        x0 = max(0, x0);
        y0 = max(0, y0);
        
        float fx = srcX - floorf(srcX);
        float fy = srcY - floorf(srcY);
        
        float v00 = input[y0 * inWidth + x0];
        float v01 = input[y0 * inWidth + x1];
        float v10 = input[y1 * inWidth + x0];
        float v11 = input[y1 * inWidth + x1];
        
        float value = v00 * (1-fx) * (1-fy) +
                      v01 * fx * (1-fy) +
                      v10 * (1-fx) * fy +
                      v11 * fx * fy;
        
        output[y * outWidth + x] = value;
    }
    ''', 'manualBilinear')
    
    block = (16, 16)
    grid = ((out_size + 15) // 16, (out_size + 15) // 16)
    
    # Warmup
    manual_bilinear(grid, block, (d_input, d_output, out_size, out_size, in_size, in_size))
    cp.cuda.Stream.null.synchronize()
    
    # Benchmark
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    for _ in range(10):
        manual_bilinear(grid, block, (d_input, d_output, out_size, out_size, in_size, in_size))
    end.record()
    end.synchronize()
    
    manual_time = cp.cuda.get_elapsed_time(start, end) / 10
    
    print(f"\n  Manual Bilinear: {manual_time:.3f} ms")
    print("  (Texture-based would be ~2-3x faster with hardware filtering)")


def demonstrate_address_modes():
    """Demonstrate different texture addressing modes."""
    print("\n" + "="*70)
    print("TEXTURE ADDRESSING MODES")
    print("="*70)
    
    # Create simple 4x4 test image
    image = np.array([
        [0.1, 0.2, 0.3, 0.4],
        [0.5, 0.6, 0.7, 0.8],
        [0.9, 1.0, 1.1, 1.2],
        [1.3, 1.4, 1.5, 1.6]
    ], dtype=np.float32)
    
    print("\n  Original 4x4 image:")
    print(image)
    
    # Simulate different addressing modes using numpy
    def clamp_address(x, y, h, w):
        return (max(0, min(y, h-1)), max(0, min(x, w-1)))
    
    def wrap_address(x, y, h, w):
        return (y % h, x % w)
    
    def mirror_address(x, y, h, w):
        y = y % (2 * h)
        x = x % (2 * w)
        if y >= h: y = 2 * h - 1 - y
        if x >= w: x = 2 * w - 1 - x
        return (y, x)
    
    print("\n  Accessing position (5, 2) with different modes:")
    print(f"    Clamp mode: {image[clamp_address(5, 2, 4, 4)]}")
    print(f"    Wrap mode:  {image[wrap_address(5, 2, 4, 4)]}")
    print(f"    Mirror mode: {image[mirror_address(5, 2, 4, 4)]}")


# ============================================================================
# SURFACE MEMORY OPERATIONS
# ============================================================================

def demonstrate_surface_memory():
    """Demonstrate surface memory for read-write image operations."""
    print("\n" + "="*70)
    print("SURFACE MEMORY DEMONSTRATION")
    print("="*70)
    
    # Surface memory allows read-write access to CUDA arrays
    # In CuPy, we can simulate this with regular arrays
    
    width, height = 512, 512
    
    # Create input image
    image = np.random.rand(height, width).astype(np.float32)
    d_image = cp.asarray(image)
    
    # In-place image operation (what surface memory enables)
    inplace_kernel = cp.RawKernel(r'''
    extern "C" __global__
    void inplaceTransform(float* image, int width, int height) {
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        
        if (x >= width || y >= height) return;
        
        int idx = y * width + x;
        
        // Read current value
        float val = image[idx];
        
        // Transform in-place
        image[idx] = 1.0f - val;  // Invert
    }
    ''', 'inplaceTransform')
    
    block = (16, 16)
    grid = ((width + 15) // 16, (height + 15) // 16)
    
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    
    start.record()
    inplace_kernel(grid, block, (d_image, width, height))
    end.record()
    end.synchronize()
    
    time_ms = cp.cuda.get_elapsed_time(start, end)
    
    print(f"  In-place image inversion: {time_ms:.3f} ms")
    print(f"  Image size: {width}x{height}")
    
    # Verify
    result = d_image.get()
    expected = 1.0 - image
    max_error = np.max(np.abs(result - expected))
    print(f"  Max error: {max_error:.2e}")


# ============================================================================
# PERFORMANCE ANALYSIS
# ============================================================================

def performance_analysis():
    """Comprehensive performance analysis of texture operations."""
    print("\n" + "="*70)
    print("PERFORMANCE ANALYSIS")
    print("="*70)
    
    sizes = [256, 512, 1024, 2048, 4096]
    kernel_sizes = [3, 5, 7, 9]
    
    print("\nConvolution Performance (Global Memory):")
    print(f"{'Size':<10} {'Kernel':<10} {'Time (ms)':<12} {'Bandwidth (GB/s)':<15}")
    print("-" * 50)
    
    processor = TextureImageProcessor()
    
    for size in sizes:
        for ks in kernel_sizes:
            if size >= 4096 and ks >= 7:
                continue  # Skip very large combinations
            
            image = np.random.rand(size, size).astype(np.float32)
            kernel = np.random.rand(ks, ks).astype(np.float32)
            kernel /= kernel.sum()
            
            result = processor.convolve_global(image, kernel)
            
            # Calculate bandwidth
            reads = size * size * 4 * ks * ks  # Reads per pixel
            writes = size * size * 4
            bandwidth = (reads + writes) / (result.time_ms / 1000) / 1e9
            
            print(f"{size}x{size:<5} {ks}x{ks:<6} {result.time_ms:<12.3f} {bandwidth:<15.1f}")


def main():
    """Main function running all demonstrations."""
    print("="*70)
    print("DAY 12: TEXTURE AND SURFACE MEMORY")
    print("Phase 6: AI/ML Platform Engineering with GPU Programming")
    print("="*70)
    
    print(f"\nGPU: {cp.cuda.Device().name}")
    
    demonstrate_texture_vs_global()
    demonstrate_bilinear_interpolation()
    demonstrate_address_modes()
    demonstrate_surface_memory()
    performance_analysis()
    
    print("\n" + "="*70)
    print("Day 12 Complete: Texture and Surface Memory")
    print("="*70)


if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: "GPU Image Processing Pipeline"

### Lab Objectives
1. Build a complete image processing pipeline
2. Implement multiple filters using texture concepts
3. Measure performance improvements

### Implementation

```python
#!/usr/bin/env python3
"""
Lab: GPU Image Processing Pipeline
Day 12: Texture and Surface Memory
"""

import cupy as cp
import numpy as np
from PIL import Image
from typing import List, Callable
from dataclasses import dataclass


@dataclass
class PipelineStage:
    """A stage in the image processing pipeline."""
    name: str
    kernel: cp.RawKernel
    params: dict


class ImageProcessingPipeline:
    """
    GPU-accelerated image processing pipeline.
    
    This demonstrates how to chain multiple image operations
    efficiently on the GPU using texture-optimized access patterns.
    """
    
    def __init__(self):
        self.stages: List[PipelineStage] = []
        self._compile_kernels()
    
    def _compile_kernels(self):
        """Compile all pipeline kernels."""
        
        # Brightness/Contrast adjustment
        self.brightness_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void adjustBrightness(
            const float* input, float* output,
            int width, int height,
            float brightness, float contrast
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            int idx = y * width + x;
            float val = input[idx];
            
            // Apply contrast then brightness
            val = (val - 0.5f) * contrast + 0.5f + brightness;
            
            // Clamp to [0, 1]
            output[idx] = fminf(1.0f, fmaxf(0.0f, val));
        }
        ''', 'adjustBrightness')
        
        # Gaussian blur
        self.blur_kernel = cp.RawKernel(r'''
        __constant__ float gaussKernel[25] = {
            0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f,
            0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f,
            0.023438f, 0.093750f, 0.140625f, 0.093750f, 0.023438f,
            0.015625f, 0.062500f, 0.093750f, 0.062500f, 0.015625f,
            0.003906f, 0.015625f, 0.023438f, 0.015625f, 0.003906f
        };
        
        extern "C" __global__
        void gaussianBlur(
            const float* input, float* output,
            int width, int height
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            float sum = 0.0f;
            
            for (int ky = -2; ky <= 2; ky++) {
                for (int kx = -2; kx <= 2; kx++) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    
                    sum += input[iy * width + ix] * 
                           gaussKernel[(ky + 2) * 5 + (kx + 2)];
                }
            }
            
            output[y * width + x] = sum;
        }
        ''', 'gaussianBlur')
        
        # Edge detection
        self.edge_kernel = cp.RawKernel(r'''
        extern "C" __global__
        void detectEdges(
            const float* input, float* output,
            int width, int height
        ) {
            int x = blockIdx.x * blockDim.x + threadIdx.x;
            int y = blockIdx.y * blockDim.y + threadIdx.y;
            
            if (x >= width || y >= height) return;
            
            // Sobel operators
            float gx = 0.0f, gy = 0.0f;
            
            int sobelX[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
            int sobelY[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};
            
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int ix = min(max(x + kx, 0), width - 1);
                    int iy = min(max(y + ky, 0), height - 1);
                    
                    float val = input[iy * width + ix];
                    gx += val * sobelX[ky + 1][kx + 1];
                    gy += val * sobelY[ky + 1][kx + 1];
                }
            }
            
            output[y * width + x] = sqrtf(gx * gx + gy * gy);
        }
        ''', 'detectEdges')
    
    def add_brightness(self, brightness: float = 0.0, contrast: float = 1.0):
        """Add brightness/contrast adjustment stage."""
        self.stages.append(PipelineStage(
            name=f"Brightness({brightness}, {contrast})",
            kernel=self.brightness_kernel,
            params={'brightness': brightness, 'contrast': contrast}
        ))
        return self
    
    def add_blur(self):
        """Add Gaussian blur stage."""
        self.stages.append(PipelineStage(
            name="GaussianBlur",
            kernel=self.blur_kernel,
            params={}
        ))
        return self
    
    def add_edge_detection(self):
        """Add edge detection stage."""
        self.stages.append(PipelineStage(
            name="EdgeDetection",
            kernel=self.edge_kernel,
            params={}
        ))
        return self
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through the pipeline.
        
        Args:
            image: Input image (H, W) float32 in [0, 1]
            
        Returns:
            Processed image (H, W) float32
        """
        height, width = image.shape
        
        # Transfer to GPU
        d_input = cp.asarray(image)
        d_output = cp.zeros_like(d_input)
        
        block = (16, 16)
        grid = ((width + 15) // 16, (height + 15) // 16)
        
        current = d_input
        
        for stage in self.stages:
            if stage.name.startswith("Brightness"):
                stage.kernel(grid, block, (
                    current, d_output, width, height,
                    stage.params['brightness'], stage.params['contrast']
                ))
            else:
                stage.kernel(grid, block, (current, d_output, width, height))
            
            # Swap buffers
            current, d_output = d_output, current
        
        cp.cuda.Stream.null.synchronize()
        
        return current.get()
    
    def benchmark(self, image: np.ndarray, iterations: int = 100) -> dict:
        """Benchmark the pipeline."""
        height, width = image.shape
        
        d_input = cp.asarray(image)
        d_temp = cp.zeros_like(d_input)
        
        # Warmup
        for _ in range(5):
            _ = self.process(image)
        
        # Benchmark
        start = cp.cuda.Event()
        end = cp.cuda.Event()
        
        start.record()
        for _ in range(iterations):
            _ = self.process(image)
        end.record()
        end.synchronize()
        
        total_time = cp.cuda.get_elapsed_time(start, end)
        
        return {
            'total_time_ms': total_time,
            'avg_time_ms': total_time / iterations,
            'fps': iterations / (total_time / 1000),
            'stages': len(self.stages)
        }


def run_lab():
    """Run the image processing lab."""
    print("="*60)
    print("LAB: GPU Image Processing Pipeline")
    print("Day 12: Texture and Surface Memory")
    print("="*60)
    
    # Create test image
    size = 1024
    image = np.random.rand(size, size).astype(np.float32)
    
    print(f"\nImage size: {size}x{size}")
    
    # Test different pipelines
    pipelines = [
        ("Blur only", ImageProcessingPipeline().add_blur()),
        ("Edge only", ImageProcessingPipeline().add_edge_detection()),
        ("Blur + Edge", ImageProcessingPipeline().add_blur().add_edge_detection()),
        ("Full pipeline", ImageProcessingPipeline()
            .add_brightness(0.1, 1.2)
            .add_blur()
            .add_edge_detection()),
    ]
    
    print("\nPipeline Performance:")
    print(f"{'Pipeline':<20} {'Stages':<8} {'Time (ms)':<12} {'FPS':<10}")
    print("-" * 50)
    
    for name, pipeline in pipelines:
        result = pipeline.benchmark(image, 100)
        print(f"{name:<20} {result['stages']:<8} "
              f"{result['avg_time_ms']:<12.3f} {result['fps']:<10.1f}")
    
    print("\nLab complete!")


if __name__ == "__main__":
    run_lab()
```

---

## 📝 Daily Summary

### Key Takeaways
1. **Texture memory** is optimized for 2D spatial locality
2. **Hardware interpolation** (bilinear) is essentially "free"
3. **Addressing modes** handle boundaries automatically
4. **CUDA arrays** provide better 2D caching than linear memory
5. **Surface memory** adds write capability to texture features
6. **Best for image processing** and spatial data access patterns

### When to Use Texture Memory
✅ **Good Use Cases:**
- Image processing (convolution, filtering)
- Volume rendering
- Lookup tables
- Any 2D spatially-local access

❌ **Avoid When:**
- Linear memory access patterns
- Write-heavy workloads
- Non-spatial data

### API Summary
```cpp
// Texture object creation
cudaResourceDesc resDesc;
cudaTextureDesc texDesc;
cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL);

// Texture sampling
float value = tex2D<float>(texObj, x, y);

// Surface read/write
surf2Dread(&value, surfObj, x * sizeof(float), y);
surf2Dwrite(value, surfObj, x * sizeof(float), y);
```

---

**Day 12 Complete** ✅

*Next: Day 13 - Multi-GPU Programming - Scaling beyond a single GPU!*

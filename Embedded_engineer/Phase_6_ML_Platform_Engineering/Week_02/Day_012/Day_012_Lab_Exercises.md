# Day 12: Texture and Surface Memory - Lab Exercises
### Phase 6: AI/ML Platform Engineering with GPU Programming

---

## 🔬 Exercise 1: Understanding Texture Memory (Beginner)

### Objective
Learn when texture memory provides benefits.

### Code

```python
import torch
import torch.nn.functional as F

def texture_benefits_demo():
    """Demonstrate texture-like operations in PyTorch."""
    
    print("Texture Memory Benefits")
    print("=" * 40)
    
    # Texture memory excels at:
    # 1. 2D spatial locality (images)
    # 2. Interpolation (bilinear, bicubic)
    # 3. Boundary handling (clamp, wrap)
    
    # Create sample image
    image = torch.rand(1, 3, 256, 256, device='cuda')
    
    # Resize with interpolation (uses texture-like caching)
    resized = F.interpolate(image, size=(512, 512), mode='bilinear', align_corners=False)
    
    print(f"Input shape: {image.shape}")
    print(f"Output shape: {resized.shape}")
    
    # When texture memory helps:
    print("\nTexture memory benefits when:")
    print("  ✓ Reading 2D data with spatial locality")
    print("  ✓ Need hardware interpolation")
    print("  ✓ Access patterns are not coalesced")
    print("  ✓ Read-only data")

if __name__ == "__main__":
    texture_benefits_demo()
```

---

## 🔬 Exercise 2: Image Processing with Spatial Access (Intermediate)

### Objective
Compare access patterns for image processing.

### Code

```python
import torch
import time

def compare_access_patterns():
    """Compare row-major vs block-based access."""
    
    H, W = 4096, 4096
    image = torch.rand(H, W, device='cuda')
    
    print("Image Access Pattern Comparison")
    print("=" * 40)
    
    # Pattern 1: Row-major (good for global memory)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        output1 = image.clone()
    torch.cuda.synchronize()
    row_time = (time.perf_counter() - start) / 10 * 1000
    
    # Pattern 2: Column-major (poor coalescing)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        output2 = image.T.clone()  # Transpose forces column access
    torch.cuda.synchronize()
    col_time = (time.perf_counter() - start) / 10 * 1000
    
    # Pattern 3: Random 2D access (texture would help here)
    indices_y = torch.randint(0, H, (H*W//4,), device='cuda')
    indices_x = torch.randint(0, W, (H*W//4,), device='cuda')
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(10):
        output3 = image[indices_y, indices_x]
    torch.cuda.synchronize()
    rand_time = (time.perf_counter() - start) / 10 * 1000
    
    print(f"Row-major access:    {row_time:.2f} ms")
    print(f"Column-major access: {col_time:.2f} ms")
    print(f"Random 2D access:    {rand_time:.2f} ms")
    print(f"\nTexture memory would help most with random 2D access")

if __name__ == "__main__":
    compare_access_patterns()
```

---

## 🔬 Exercise 3: Image Filtering (Advanced)

### Objective
Implement convolution-like operation optimized for 2D access.

### Code

```python
import torch
import torch.nn.functional as F
import time

def optimized_convolution():
    """Compare convolution implementations."""
    
    batch = 32
    channels = 64
    H, W = 256, 256
    kernel_size = 3
    
    # Input image
    x = torch.rand(batch, channels, H, W, device='cuda')
    
    # Convolution kernel
    weight = torch.rand(128, channels, kernel_size, kernel_size, device='cuda')
    
    print("Convolution Optimization")
    print("=" * 40)
    
    # Warm up
    _ = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()
    
    # cuDNN convolution (uses texture-like caching internally)
    start = time.perf_counter()
    for _ in range(100):
        y = F.conv2d(x, weight, padding=1)
    torch.cuda.synchronize()
    conv_time = (time.perf_counter() - start) / 100 * 1000
    
    print(f"Input: {x.shape}")
    print(f"Kernel: {weight.shape}")
    print(f"Output: {y.shape}")
    print(f"Time: {conv_time:.3f} ms")
    
    # Calculate FLOPS
    flops = 2 * batch * 128 * H * W * channels * kernel_size * kernel_size
    tflops = flops / (conv_time / 1000) / 1e12
    print(f"Performance: {tflops:.2f} TFLOPS")

if __name__ == "__main__":
    optimized_convolution()
```

---

## 🐛 Common Issues

### Issue: Texture memory not faster
**Cause:** Access pattern is already coalesced
**Fix:** Texture memory mainly helps with uncoalesced 2D access

### Issue: cuDNN not using expected algorithm
**Fix:** Use `torch.backends.cudnn.benchmark = True`

---

## 📚 Additional Resources
- [CUDA Texture Memory](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#texture-memory)
- [cuDNN Convolution](https://docs.nvidia.com/deeplearning/cudnn/developer-guide/index.html)

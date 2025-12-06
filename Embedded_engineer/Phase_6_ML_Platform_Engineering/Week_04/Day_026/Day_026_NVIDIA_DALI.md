# Day 26: High-Performance Data Loading with NVIDIA DALI
### Phase 6: AI/ML Platform Engineering with GPU Programming | Week 4: TensorRT & Inference Optimization

---

> **🎯 Focus Area:** Eliminate the "CPU Bottleneck" in Deep Learning training/inference by moving data preprocessing (JPEG decoding, resizing, augmentation) directly to the GPU using the NVIDIA Data Loading Library (DALI).

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Identify** CPU starvation issues where the GPU waits for Preprocessing (`0%` volatile utilization spikes).
2.  **Construct** a DALI Pipeline to decode and augment images on the GPU.
3.  **Integrate** DALI iterators with PyTorch models.
4.  **Benchmark** data throughput (images/second) of DALI vs. standard `DataLoader`.
5.  **Implement** a mixed pipeline (CPU-Disk I/O -> GPU-Compute).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (DALI optimizes for nvJPEG and GPU memory copies).

### Software Environment
```bash
# Install NVIDIA DALI (CUDA 12.x version)
pip install nvidia-dali-cuda120
```

### Prior Knowledge
- PyTorch DataLoaders (`Dataset`, `DataLoader`, `num_workers`).
- Image Processing basics (Resizing, Normalization).

---

## 📖 Theoretical Foundation

### 1. The CPU Bottleneck

In standard Deep Learning:
1.  **CPU** reads JPEG from disk.
2.  **CPU** decodes JPEG to RGB.
3.  **CPU** resizes/crops/normalizes.
4.  **CPU** casts to Float Tensor.
5.  **PCIe Bus** copies Tensor to GPU.
6.  **GPU** trains network.

As GPUs get faster (A100, H100), steps 1-4 become the bottleneck. The GPU sits idle waiting for data.

### 2. DALI Architecture

DALI (Data Loading Library) defines a **Graph** of operations, executed asynchronously.
*   **nvJPEG:** Decodes images *on the GPU*.
*   **Fused Kernels:** Resizing and Normalization happen in highly optimized CUDA kernels.
*   **Direct Memory Access:** Reduced copying between Host and Device.

### 3. The Pipeline

Programming DALI involves defining a `Pipeline` class:
1.  **Source:** Disk (`fn.readers.file`), Video (`fn.readers.video`), or memory.
2.  **Ops:** `fn.decoders.image`, `fn.resize`, `fn.rotate`.
3.  **Output:** Batches of GPU tensors ready for consumption.

---

## 💻 Implementation

### 👨‍💻 Core Implementation: Basic DALI Pipeline

This script creates a pipeline that reads images, decodes them on GPU, and resizes them.

#### 📁 `src/dali_basic.py`
```python
#!/usr/bin/env python3
"""
Day 26: Basic DALI Pipeline
Phase 6: Platform Engineering
"""

from nvidia.dali import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.pytorch import DALIGenericIterator
import os
import torch
import time

# Helper to create dummy data
def create_dummy_data(root="dummy_data", num=1000):
    import numpy as np
    from PIL import Image
    if not os.path.exists(root):
        os.makedirs(root)
    
    print(f"Generating {num} dummy JPEGs...")
    for i in range(num):
        # Random noise image
        img = np.random.randint(0, 255, (500, 500, 3), dtype=np.uint8)
        Image.fromarray(img).save(os.path.join(root, f"{i}.jpg"))

# 1. Define the Pipeline
@pipeline_def
def get_dali_pipeline(data_dir):
    # a. Read from disk (Execution: CPU)
    jpegs, labels = fn.readers.file(
        file_root=data_dir,
        random_shuffle=True,
        name="Reader"
    )
    
    # b. Decode to RGB (Execution: Mixed/GPU)
    # device='mixed' means Input on CPU, Output on GPU
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    # c. Resize to 224x224 (Execution: GPU)
    images = fn.resize(
        images,
        resize_x=224,
        resize_y=224,
        interp_type=types.INTERP_LINEAR
    )
    
    # d. Normalize (Execution: GPU)
    # Mean/Std for ImageNet
    images = fn.crop_mirror_normalize(
        images,
        mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
        std=[0.229 * 255, 0.224 * 255, 0.225 * 255],
        output_dtype=types.FLOAT
    )
    
    return images, labels

def main():
    root_dir = "dummy_data"
    create_dummy_data(root_dir)
    
    BATCH_SIZE = 64
    NUM_THREADS = 4
    DEVICE_ID = 0
    
    # 2. Build Pipeline
    pipe = get_dali_pipeline(
        data_dir=root_dir,
        batch_size=BATCH_SIZE,
        num_threads=NUM_THREADS,
        device_id=DEVICE_ID
    )
    pipe.build()
    
    # 3. Create PyTorch Iterator
    # DALI output is wrapping in a PyTorch-compatible iterable
    dali_iter = DALIGenericIterator(
        pipe, 
        output_map=["data", "label"],
        reader_name="Reader",
        auto_reset=True
    )
    
    print("\nStarting Benchmark...")
    start = time.time()
    
    count = 0
    for i, data in enumerate(dali_iter):
        # data is a list of dicts. 
        # data[0]['data'] is the image batch on GPU
        # data[0]['label'] is the label batch on GPU
        
        images = data[0]["data"]
        labels = data[0]["label"]
        
        # Verify it is on GPU
        if i == 0:
            print(f"Tensor Device: {images.device}")
            print(f"Tensor Shape: {images.shape}")
        
        count += BATCH
        
        # Stop after some batches
        if i > 50: 
            break
            
    end = time.time()
    total_time = end - start
    print(f"Processed {count} images in {total_time:.2f}s")
    print(f"Throughput: {count / total_time:.1f} images/sec")

if __name__ == "__main__":
    main()
```

### 👨‍💻 Comparison: PyTorch Native vs DALI

Often, DALI is unnecessary for small models (ResNet18) because the GPU compute dominates. But for **ResNet50 or EfficientNet**, DALI shines.

#### 📁 `src/benchmark_loader.py`
```python
#!/usr/bin/env python3
"""
Day 26: Benchmark DALI vs PyTorch DataLoader
"""

import time
import torch
from torchvision import datasets, transforms
import os

# Assume dummy_data exists from previous script source

def benchmark_pytorch(root_dir, batch_size):
    print("Benchmarking PyTorch DataLoader (num_workers=4)...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Use ImageFolder (requires subdirectory structure, assume dummy_data works or fix it)
    # create dummy struct
    # dummy_data/class1/img.jpg
    
    dataset = datasets.FakeData(size=1000, image_size=(3, 500, 500), transform=transform)
    
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True # Optimization for transfer
    )
    
    start = time.time()
    count = 0
    
    for i, (imgs, labels) in enumerate(loader):
        imgs = imgs.cuda(non_blocking=True)
        count += batch_size
        if i > 50: break
        
    end = time.time()
    print(f"PyTorch Throughput: {count / (end - start):.1f} img/sec")

# Run DALI benchmark again here for comparison...
# (Omitted reuse of previous code for brevity)

if __name__ == "__main__":
    benchmark_pytorch("dummy_data", 64)
    # On high-end CPUs, PyTorch is fast. 
    # On generic cloud vCPUs + Powerful GPU, DALI usually wins by 2x-4x.
```

---

## 🔬 Lab Exercise: "Video Pipeline"

### Lab Objectives
1.  Video processing is incredibly heavy (decoding H.264).
2.  Use `nvidia.dali.fn.readers.video`.
3.  Build a pipeline that reads a video file, extracts sequences of 16 frames, resizes them, and streams to GPU.

### Key Snippet
```python
video = fn.readers.video(
    device="gpu", # GPU Decoding of Video!
    filenames=["my_video.mp4"],
    sequence_length=16,
    shard_id=0,
    num_shards=1
)
```
This is essential for "Action Recognition" models (3D CNNs).

---

## 📝 Daily Summary

### Key Takeaways
1.  **Decode on GPU:** The heaviest part of ETL is usually Jpeg/Video decoding. `nvJPEG`/`nvDEC` hardware blocks on NVIDIA GPUs handle this almost for free.
2.  **Pipeline Def:** DALI uses a declarative graph definition, similar to TensorFlow graphs or Onyx, separate from Python execution.
3.  **Use Cases:** DALI is critical for computer vision. For NLP (text), the bottleneck is usually disk I/O or tokenization (which runs on CPU), so DALI is less common there (though support exists).
4.  **Integration:** DALI works seamless with PyTorch, TensorFlow, PaddlePaddle, and JAX.

### API Summary
```python
@pipeline_def
def my_pipe():
    # Reader
    data, label = fn.readers.file(...)
    # Ops
    data = fn.decoders.image(data, device="mixed")
    data = fn.resize(data, size=(224,224))
    return data, label

# Iterator
iter = DALIGenericIterator(pipe, ["data", "label"])
```

---

**Day 26 Complete** ✅

*Next: Day 27 - Profiling & Optimization Tools - Investigating bottlenecks deep inside the GPU with Nsight Systems.*

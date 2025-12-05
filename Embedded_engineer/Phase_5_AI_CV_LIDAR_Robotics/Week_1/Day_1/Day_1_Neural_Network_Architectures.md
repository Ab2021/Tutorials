# Day 1: Neural Network Architectures for Robotics
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> This document provides a comprehensive deep dive into neural network architectures specifically tailored for robotics applications. It covers the evolution from CNNs to Vision Transformers, hybrid architectures, and real-time inference considerations.
> - **Target Length:** 1200+ lines
> - **Focus:** Mathematical foundations, architectural details, and PyTorch implementations.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Differentiate** between CNNs (CovNets) and Vision Transformers (ViTs) in the context of robotic perception.
2.  **Implement** modern architectures like ConvNeXt and Swin Transformer from scratch in PyTorch.
3.  **Analyze** the latency-accuracy trade-offs of different backbones (EfficientNet, MobileNet, ResNet) for edge deployment.
4.  **Design** a hybrid architecture that leverages the strengths of both local features (CNNs) and global context (Transformers).
5.  **Benchmark** model inference speeds on CPU/GPU and understand the implications for closed-loop robotic control.

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
| Component | Specification | Purpose |
|-----------|--------------|---------|
| GPU | NVIDIA RTX 3060+ or Jetson Orin | Training and Inference Benchmarking |
| CPU | Multi-core (i7/Ryzen 7) | Data loading and preprocessing |
| RAM | 16GB+ | Model loading |

### Software Environment
```bash
# Create conda environment
conda create -n robotics_ai python=3.10
conda activate robotics_ai

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install auxiliary libraries
pip install timm==0.9.12  # PyTorch Image Models
pip install torchinfo     # Model summary
pip install matplotlib seaborn numpy pandas
pip install fvcore        # FLOPs counting
```

### Prior Knowledge
- Basic understanding of Linear Algebra (Matrix multiplications, Dot products).
- Familiarity with PyTorch `nn.Module` and the training loop.
- Understanding of basic CNN concepts (Convolution, Pooling, Activation).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Evolution of Visual Backbones

For a robot to interact with the world, it must first "see" and understand it. The visual backbone is the feature extractor engine that powers tasks like Object Detection, Segmentation, and Pose Estimation.

#### 1.1 Convolutional Neural Networks (CNNs) Revisited

CNNs have been the workhorse of computer vision for a decade. Their design is inspired by the biological visual cortex, leveraging two key inductive biases:
1.  **Translation Equivariance:** If an object shifts in the image, the feature map representation shifts equivalently.
2.  **Locality:** Pixels close to each other are highly correlated.

**Mathematical Formulation of 2D Convolution:**
Given an input image $I \in \mathbb{R}^{H \times W \times C}$ and a kernel $K \in \mathbb{R}^{k \times k \times C \times C_{out}}$, the output feature map $O$ at position $(i, j)$ is:

$$O(i, j) = \sum_{m=0}^{k-1} \sum_{n=0}^{k-1} I(i+m, j+n) \cdot K(m, n) + b$$

**Key Architectures for Robotics:**

*   **ResNet (2015):** Introduced skip connections ($y = F(x) + x$) to solve the vanishing gradient problem, allowing deeper networks (50, 101, 152 layers).
    *   *Robotics Relevance:* Standard backbone for many reliable detection systems (e.g., Mask R-CNN). The stride-32 output is standard.
*   **MobileNetV2/V3 (2018/2019):** Introduced Inverted Residuals and Linear Bottlenecks using Depthwise Separable Convolutions.
    *   *Depthwise Conv:* Applies a single filter per input channel. Complexity reduces from $H W C_{in} C_{out} k^2$ to $H W C_{in} (k^2 + C_{out})$.
    *   *Robotics Relevance:* Crucial for battery-powered, compute-constrained robots (drones, quadruped payload).
*   **EfficientNet (2019):** Used Compound Scaling to balance Depth, Width, and Resolution ($\alpha, \beta, \gamma$) via Neural Architecture Search (NAS).
*   **ConvNeXt (2022):** A "modernized" ResNet that adopts Transformer design choices (larger kernels $7 \times 7$, LayerNorm, GELU, fewer activations) to compete with ViTs while staying purely convolutional.

#### 1.2 Vision Transformers (ViTs)

Transformers processes images as sequences of patches, removing the strict locality inductive bias in favor of **Global Context** via Self-Attention.

**The Self-Attention Mechanism:**
Given a sequence of patch embeddings $X \in \mathbb{R}^{N \times D}$:

1.  **Projections:** Query $Q = XW_Q$, Key $K = XW_K$, Value $V = XW_V$.
2.  **Attention Score:**
    $$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Where $\sqrt{d_k}$ is the scaling factor to prevent vanishing gradients in softmax for large dimensions.

**ViT Architecture:**
1.  **Patchify:** Split image into $P \times P$ patches (e.g., $16 \times 16$).
2.  **Linear Projection:** Flatten patches and project to embedding dimension $D$.
3.  **CLS Token:** Prepend a learnable class token.
4.  **Positional Embedding:** Add learnable position vectors (since attention is permutation invariant).
5.  **Transformer Encoder:** Stack of Multi-Head Self-Attention (MSA) and MLP blocks.

**Robotics Relevance:**
*   **Pros:** Better performance on large datasets; captures long-range dependencies (useful for scene understanding, e.g., relating a door handle to the door hinge across the image).
*   **Cons:** Quadratic complexity $O(N^2)$ with respect to image size (since number of patches $N = \frac{HW}{P^2}$). Harder to optimize for edge hardware (TensorRT support is improving but CNNs are still faster).

#### 1.3 Hierarchical Transformers (Swin)

Standard ViT produces a single-scale feature map (isotropic). Dense prediction tasks (Segmentation/Detection) require multi-scale features (FPN).

**Swin Transformer:** Introduces **Shifted Window Attention**.
*   Computes attention only within local windows (linear complexity $O(N)$).
*   Shifts windows in consecutive layers to enable cross-window connection.
*   produces hierarchical feature maps ($\frac{H}{4}, \frac{H}{8}, \frac{H}{16}, \frac{H}{32}$), making it a drop-in replacement for ResNet.

### 🔹 Part 2: Hybrid Architectures & Design Choices

In robotics, we often need the "best of both worlds":
*   **CNNs:** Low-level edge/texture detection, translation invariance, hardware efficiency.
*   **Transformers:** Semantic reasoning, global context, multi-modal fusion capability.

**Hybrid Designs:**
1.  **Convolutional Stem:** Use Conv layers for the first few stages to reduce resolution (handling high-res sensor input efficiently) and then use Transformer blocks for high-level reasoning.
    *   *Example:* LeViT, CoAtNet.
2.  **Parallel Branches:** Maintain a high-res Conv branch and a low-res Attn branch (e.g., HRNet variants).

**Real-Time Inference Considerations:**
*   **Throughput (FPS):** Critical for high-speed motion (e.g., drone flight).
*   **Latency (ms):** Critical for closed-loop control stability. A high-throughput batch processing system might still have high latency per frame.
*   **Jitter:** Variability in latency. Real-time systems prefer consistent latency over average latency.

---

## 💻 Implementation: Building Modern Backbones

We will focus on implementing the core building blocks of **ConvNeXt** and **Swin Transformer** to understand the modern architectural choices.

### 🛠️ Project Structure
```text
day1_architectures/
├── models/
│   ├── __init__.py
│   ├── convnext.py
│   ├── vision_transformer.py
│   └── swin_transformer.py
├── utils/
│   ├── benchmark.py
│   └── flopp_counter.py
├── train.py
└── visualize_attention.py
```

### 👨‍💻 Code Implementation

#### 1. ConvNeXt Block Implementation
A modernized ResNet block. Note the use of Depthwise Conv, LayerNorm, and GELU.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):
    """
    LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering dimensions matters for performance.
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        # Depthwise Conv: 7x7 kernel, groups=dim
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) 
        self.norm = LayerNorm(dim, eps=1e-6)
        
        # Pointwise Conv 1: Expand channel dimension by 4x
        self.pwconv1 = nn.Linear(dim, 4 * dim) 
        self.act = nn.GELU()
        
        # Pointwise Conv 2: Project back to original dimension
        self.pwconv2 = nn.Linear(4 * dim, dim)
        
        # Layer Scale: Learnable scaling parameter for residual connection stability
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        # Permute for linear layers (N, C, H, W) -> (N, H, W, C)
        x = x.permute(0, 2, 3, 1) 
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        
        if self.gamma is not None:
            x = self.gamma * x
            
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)
        x = input + self.drop_path(x)
        return x

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""
    def __init__(self, drop_prob=0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output
```

#### 2. Vision Transformer Attention Block
Implementation of a standard Multi-Head Self-Attention block.

```python
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        # x shape: [Batch, Patches, Dim]
        B, N, C = x.shape
        # qkv shape: [3, Batch, Heads, Patches, Head_Dim]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        # Attention = Softmax(Q * K^T / scale)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # Output = Attention * V
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, 
                              attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
```

### 🔹 Part 3: Inference Benchmarking for Robotics

In robotics, **latency** matters more than throughput. We will build a benchmarking tool.

#### 3.1 Benchmark Utilities (`utils/benchmark.py`)

```python
import torch
import time
import numpy as np

@torch.no_grad()
def measure_latency(model, input_shape=(1, 3, 224, 224), device='cuda', warmup=10, runs=100):
    model = model.to(device)
    model.eval()
    
    dummy_input = torch.randn(input_shape).to(device)
    
    # Warmup
    print("Warming up...")
    for _ in range(warmup):
        _ = model(dummy_input)
    torch.cuda.synchronize()
    
    # Measurement
    print(f"Benchmarking with {runs} runs...")
    latencies = []
    
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    
    for _ in range(runs):
        start_event.record()
        _ = model(dummy_input)
        end_event.record()
        torch.cuda.synchronize() # Wait for GPU to finish
        latencies.append(start_event.elapsed_time(end_event)) # Time in ms
        
    latencies = np.array(latencies)
    
    stats = {
        "mean": np.mean(latencies),
        "std": np.std(latencies),
        "min": np.min(latencies),
        "max": np.max(latencies),
        "p95": np.percentile(latencies, 95),
        "p99": np.percentile(latencies, 99)
    }
    
    return stats

def print_stats(name, stats):
    print(f"--- {name} ---")
    print(f"Mean Latency: {stats['mean']:.2f} ms")
    print(f"Std Dev:      {stats['std']:.2f} ms")
    print(f"99th %%tile:   {stats['p99']:.2f} ms")
    print(f"FPS:          {1000.0 / stats['mean']:.2f}")
    print("----------------")
```

---

## 🔬 Lab Exercise: Architecture Face-off

### 1. Lab Objectives
- Compare ResNet-50, EfficientNet-B0, and ViT-Tiny on a Jetson-like constraint (if local GPU available) or standard GPU.
- Measure FLOPs and Parameters.
- Analyze which model is best for a 30Hz control loop requirement.

### 2. Step-by-Step Guide

#### Phase A: Setup Models using `timm`
We will use the `timm` library to quickly instantiate standard models.

```python
import timm
import torch
from torchinfo import summary
from utils.benchmark import measure_latency, print_stats

# Define models to benchmark
models_dict = {
    "ResNet-50": timm.create_model('resnet50', pretrained=False),
    "EfficientNet-B0": timm.create_model('efficientnet_b0', pretrained=False),
    "MobileNet-V3-Large": timm.create_model('mobilenetv3_large_100', pretrained=False),
    "ViT-Tiny": timm.create_model('vit_tiny_patch16_224', pretrained=False),
    "ConvNeXt-Tiny": timm.create_model('convnext_tiny', pretrained=False),
    "Swin-Tiny": timm.create_model('swin_tiny_patch4_window7_224', pretrained=False),
}

device = 'cuda' if torch.cuda.is_available() else 'cpu'
input_shape = (1, 3, 224, 224) # Standard ImageNet size
```

#### Phase B: Parameter and FLOPs Analysis

```python
print(f"Benchmarking on {device.upper()}")

for name, model in models_dict.items():
    print(f"\nAnalyzing {name}...")
    model_stats = summary(model, input_size=input_shape, verbose=0)
    print(f"Params: {model_stats.total_params / 1e6:.2f} M")
    print(f"Mult-Adds (FLOPs): {model_stats.total_mult_adds / 1e9:.2f} G")
    
    # Run Latency Benchmark
    stats = measure_latency(model, input_shape, device=device)
    print_stats(name, stats)
```

#### Phase C: Analysis Questions
Run the above script and answer:
1.  **Efficiency:** Which model provides the best trade-off between Parameters and Latency? (Likely MobileNet or EfficientNet).
2.  **Transformer Overhead:** Compare ViT-Tiny vs ResNet-50. Even if ViT-Tiny has fewer parameters, does it run faster? (Often no, due to memory access patterns and lack of inductive bias).
3.  **Jitter:** Look at the 99th percentile latency. Which architecture is more deterministic?

### 3. Expected Output (Sample on RTX 3060)
```text
Analyzing ResNet-50...
Params: 25.56 M
Mult-Adds (FLOPs): 4.12 G
--- ResNet-50 ---
Mean Latency: 4.20 ms
FPS:          238.10
----------------

Analyzing ViT-Tiny...
Params: 5.72 M
Mult-Adds (FLOPs): 1.10 G
--- ViT-Tiny ---
Mean Latency: 5.10 ms
FPS:          196.08
----------------
```
*Note: Despite lower FLOPs, ViT might be slower due to unoptimized attention kernels compared to cuDNN convolutions.*

---

## 🧪 Advanced Lab: Implementing Flash Attention

> **Challenge:** Transformers are memory hungry. Implement a basic version of Flash Attention (tiling) or use PyTorch 2.0's scaled_dot_product_attention to see the speedup.

### Code Snippet: PyTorch 2.0 Scaled Dot Product Attention
```python
import torch.nn.functional as F

class FlashAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # QKV projection...

    def forward(self, x):
        # ... (Get q, k, v) ...
        # Standard:
        # attn = (q @ k.transpose(-2, -1)) * scale
        # attn = attn.softmax(dim=-1)
        # x = attn @ v

        # Flash Attention (Memory Effiecient):
        x = F.scaled_dot_product_attention(q, k, v, is_causal=False)
        # ... output projection
        return x
```
**Task:** Replace the `Attention` class in the `ViTBlock` above with one using `F.scaled_dot_product_attention` and benchmark the difference in `max_memory_reserved` and latency.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Shape Mismatch in ViT
*   **Symptom:** `RuntimeError: mat1 and mat2 shapes cannot be multiplied` during linear projection.
*   **Cause:** The flattened patch size must match the embedding input. $P \times P \times C$ must equal the Linear layer input dim.
*   **Fix:** Ensure `patch_size * patch_size * in_chans == embed_dim`.

#### 2. CUDA Out of Memory (OOM)
*   **Symptom:** `RuntimeError: CUDA out of memory`.
*   **Cause:** Transformers store huge attention matrices ($N \times N$). For high-res images, $N$ grows quadratically.
*   **Fix:**
    *   Reduce Batch Size.
    *   Use Mixed Precision (`torch.cuda.amp`).
    *   Use Swin Transformer (Linear complexity).

#### 3. LayerNorm Instability
*   **Symptom:** Loss assumes NaN during training.
*   **Cause:** Deep Transformers are hard to train. Large gradients can cause instability.
*   **Fix:**
    *   Use `LayerScale` (as in ConvNeXt).
    *   Use Warmup cosine Learning Rate schedule.
    *   Check `eps` in LayerNorm.

---

## ⚡ Optimization & Best Practices

### 1. Use "Channels Last" Memory Format
PyTorch convolutions run faster in `channels_last` (NHWC) format on Tensor Cores.
```python
model = model.to(memory_format=torch.channels_last)
input = input.to(memory_format=torch.channels_last)
```

### 2. Torch.compile (PyTorch 2.x)
Just-in-Time compilation to fuse kernels.
```python
import torch
model = timm.create_model('resnet50')
optimized_model = torch.compile(model)
```
*Note:* This can speed up inference by 30-50% on modern GPUs by reducing Python overhead and fusing element-wise operations.

### 3. Precision Matters
For robotics inference, FP16 or BF16 is usually sufficient and 2x faster.
```python
with torch.autocast(device_type='cuda', dtype=torch.float16):
    output = model(input)
```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** Why does a Convolutional layer have fewer parameters than a Fully Connected layer for the same input size?
    *   **A:** Weight sharing (kernel slides across image) and sparse connectivity (local receptive field).
2.  **Q:** What is the "Inductive Bias" of a ViT compared to a CNN?
    *   **A:** ViTs have much less inductive bias (no translation invariance or strict locality by default), forcing them to learn these properties from large data, but allowing more flexibility.
3.  **Q:** Why are "Shifted Windows" used in Swin Transformer?
    *   **A:** To allow information exchange between non-overlapping local windows, enabling global context build-up over layers while keeping compute linear.

### Challenge Task
> **Task:** Combine a ResNet-18 stem (first 3 layers) with a 2-layer Transformer Encoder.
> 1. Pass image through ResNet stem -> features.
> 2. Flatten features -> sequence.
> 3. Add Positional Embeddings.
> 4. Pass through Transformer.
> **Goal:** Create a valid `forward()` pass for this Hybrid architecture.

---

## 🚀 Project: NASA-JPL Open-Source Rover Perception

**Context:** The NASA Open Source Rover is a DIY version of Curiosity. We need a perception module to classify terrain (Sand, Rock, Gravel) to adjust wheel traction control.

### Module Interface (`perception_module.py`)

```python
import torch
import torchvision.transforms as T
from PIL import Image

class TerrainClassifier:
    def __init__(self, model_name='mobilenetv3_large_100', weights_path=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # We use MobileNetV3 for low latency on Raspberry Pi / Jetson
        self.model = timm.create_model(model_name, pretrained=False, num_classes=3)
        if weights_path:
            self.model.load_state_dict(torch.load(weights_path))
        
        self.model.to(self.device).eval()
        
        self.transforms = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.labels = ['Sand', 'Rock', 'Gravel']

    def predict(self, image_path_or_array):
        """
        Args:
            image_path_or_array: Path to image or numpy array (H, W, 3)
        Returns:
            class_name (str), confidence (float)
        """
        if isinstance(image_path_or_array, str):
            img = Image.open(image_path_or_array).convert('RGB')
        else:
            img = Image.fromarray(image_path_or_array)
            
        input_tensor = self.transforms(img).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits = self.model(input_tensor)
            probs = F.softmax(logits, dim=1)
            
        conf, idx = probs.max(1)
        return self.labels[idx.item()], conf.item()

# Usage Mockup
if __name__ == "__main__":
    classifier = TerrainClassifier()
    # In a real loop, this comes from the camera
    terrain, conf = classifier.predict("test_rock.jpg")
    print(f"Detected Terrain: {terrain} ({conf*100:.1f}%)")
    
    if terrain == 'Sand':
        print("Action: Increase Torque, Inflate Wheels")
    elif terrain == 'Rock':
        print("Action: Decrease Speed, Enable Suspension limit")
```

---

## 📚 Further Reading & References
- **ResNet:** He et al., "Deep Residual Learning for Image Recognition" (CVPR 2016).
- **ViT:** Dosovitskiy et al., "An Image is Worth 16x16 Words" (ICLR 2021).
- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s" (CVPR 2022).
- **Swin:** Liu et al., "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows" (ICCV 2021).
- **Code:** [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)

---

**Day 1 Complete**

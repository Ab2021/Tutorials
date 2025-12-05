# Day 2: Attention Mechanisms & Transformers
## Phase 5: AI/CV/LIDAR End-to-End Robotics | Week 1: Deep Learning Foundations

---

> **📝 Content Creator Instructions:**
> This document details the inner workings of Attention Mechanisms, the engine behind the modern AI revolution.
> - **Focus:** From basic Self-Attention to efficient Flash Attention and Cross-Attention used in Sensor Fusion.
> - **Target Length:** 1200+ lines
> - **Code:** Low-level implementation of Attention heads from scratch.

---

## 🎯 Learning Objectives
*By the end of this day, the learner will be able to:*
1.  **Derive** and explain the mathematical formulation of Scaled Dot-Product Attention.
2.  **Implement** Multi-Head Self-Attention (MHSA) and Cross-Attention modules from scratch.
3.  **Visualize** attention maps to understand what part of the robot's camera feed is being focused on.
4.  **Optimize** attention for speed using Flash Attention principles.
5.  **Apply** attention mechanisms to a robotic object tracking task (TurtleBot4).

---

## 📚 Prerequisites & Preparation

### Hardware Requirements
- NVIDIA GPU (Recommended for training attention models).
- Standard CPU for inference.

### Software Environment
```bash
pip install torch torchvision
pip install einops       # Tensor manipulation
pip install bert-viz     # Visualization tool
pip install opencv-python
```

### Prior Knowledge
- Day 1: Neural Network Architectures.
- Matrix Multiplication ($ (B, N, D) \times (B, D, N) $).

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Attention Mechanism

Attention allows a model to weigh the importance of different parts of the input data differently. In robotics, this means focusing on the obstacle in front rather than the background sky.

#### 1.1 Scaled Dot-Product Attention

The core operation involves three vectors: **Query ($Q$)**, **Key ($K$)**, and **Value ($V$)**.
*   **Query:** What I am looking for.
*   **Key:** What defines the identity of the token.
*   **Value:** Only the content of the token.

**Formula:**
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$

**Why Scale by $\sqrt{d_k}$?**
For large dimensions $d_k$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients (vanishing gradients). Scaling counteracts this.

#### 1.2 Multi-Head Attention (MHA)

Instead of performing a single attention function, we project the queries, keys, and values $h$ times with different, learned linear projections. This allows the model to jointly attend to information from different representation subspaces at different positions.

Example in Robotics:
*   **Head 1:** Focuses on color/texture (e.g., red stop sign).
*   **Head 2:** Focuses on geometry/shape (e.g., octagonal shape).

$$ \text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O $$
$$ \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V) $$

#### 1.3 Positional Encodings

Since Attention is permutation invariant (unlike RNNs or CNNs), we must inject position information.

**Sinusoidal Encodings (Vaswani et al.):**
$$ PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d_{model}}) $$
$$ PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d_{model}}) $$

For images, we often use **Learnable 2D Encodings**, adding a parameter $P \in \mathbb{R}^{H \times W \times C}$ to the input embeddings.

### 🔹 Part 2: Cross-Attention & Efficiency

#### 2.1 Cross-Attention
Self-Attention uses $X$ to generate $Q, K, V$.
Cross-Attention uses one sequence $X_1$ to generate $Q$, and another $X_2$ to generate $K, V$.

**Robotics Use Case:** Sensor Fusion (BEVFusion)
*   **Query:** LiDAR Point Cloud Features.
*   **Key/Value:** Camera Image Features.
*   *Result:* Point cloud features enhanced with RGB color/texture information.

#### 2.2 Flash Attention (Efficiency)
Standard Attention computes an $N \times N$ matrix. For a 4K image ($N \approx 8M$), this is impossible.
**Flash Attention** (Dao et al.) uses **tiling** to compute attention block by block in GPU SRAM (fast cache), avoiding massive HBM (slow memory) reads/writes.
*   **Speedup:** 2-4x faster.
*   **Memory:** Linear $O(N)$ instead of Quadratic $O(N^2)$.

---

## 💻 Implementation: Attention from Scratch

We will implement a robust Attention module using `einops` for valid dimensions handling.

### 🛠️ Project Structure
```text
day2_attention/
├── layers/
│   ├── self_attention.py
│   ├── cross_attention.py
│   └── positional_encoding.py
├── visualization/
│   └── plot_attention.py
└── train_tracker.py
```

### 👨‍💻 Code Implementation

#### 1. Multi-Head Self-Attention (`layers/self_attention.py`)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        # x shape: (Batch, Seq_Len, Dim)
        b, n, _, h = *x.shape, self.heads

        # Generate Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), qkv)

        # Dot product
        # q: (b, h, n, d), k: (b, h, n, d) -> dots: (b, h, n, n)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
            mask_value = -torch.finfo(dots.dtype).max
            dots.masked_fill_(~mask, mask_value)

        # Softmax
        attn = dots.softmax(dim=-1)

        # Value aggregation
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        
        # Merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out), attn  # Return attention map for visualization
```

#### 2. Cross-Attention (`layers/cross_attention.py`)

```python
class CrossAttention(nn.Module):
    """
    Attends to 'context' (Key/Value) using 'x' (Query).
    Useful for conditioning robot actions on language instructions.
    """
    def __init__(self, dim, context_dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context, mask=None):
        # x (Query source): (Batch, Seq_x, Dim_x)
        # context (Key/Value source): (Batch, Seq_ctx, Dim_ctx)
        
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # Attention shape: (Batch, Heads, Seq_x, Seq_ctx)
        dots = torch.einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        if mask is not None:
             dots.masked_fill_(~mask, -1e9)

        attn = dots.softmax(dim=-1)
        out = torch.einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        
        return self.to_out(out), attn
```

#### 3. Positional Encoding (`layers/positional_encoding.py`)
Implementing learnable 2D positional embeddings for visual tasks.

```python
class LearnablePositionalEncoding2D(nn.Module):
    def __init__(self, dim, height, width):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, height * width, dim))
    
    def forward(self, x):
        # x: (Batch, Height*Width, Dim)
        return x + self.pos_embedding
```

---

## 🔬 Lab Exercise: Attention-Based Tracker

### 1. Lab Objectives
- Build a simple object tracker that "attends" to a target object template.
- Visualize the attention mechanism focusing on the object in a new frame.

### 2. Step-by-Step Guide

#### Phase A: The Siamese Attention Network
We will define a network that takes a `Template` (the object to track) and `Search Region` (the current camera frame) and uses Cross-Attention to find the template in the search region.

```python
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from layers.cross_attention import CrossAttention

class SiameseTracker(nn.Module):
    def __init__(self):
        super().__init__()
        # Simple CNN feature extractor (e.g., first few layers of ResNet)
        # For simplicity, we assume input is already embedded or we use a linear projection
        self.embedding = nn.Linear(3, 64) # Project RGB pixels directly (naive)
        
        self.cross_attn = CrossAttention(dim=64, context_dim=64, heads=4)
        self.localize = nn.Linear(64, 4) # Output Bounding Box [x, y, w, h]

    def forward(self, template_img, search_img):
        # Naive implementation: flatten images
        # template: (B, H, W, 3) -> (B, N, 64)
        b, h, w, c = template_img.shape
        tpl = self.embedding(template_img.reshape(b, -1, 3))
        
        b, h_s, w_s, c = search_img.shape
        search = self.embedding(search_img.reshape(b, -1, 3))
        
        # Query = Search Region, Key/Value = Template
        # "Where in the search region matches the template?"
        # Note: Usually in tracking, we use Correlation, but Attention generalizes this.
        features, attn_map = self.cross_attn(search, tpl)
        
        # Pool features to get bounding box
        pooled = features.mean(dim=1)
        bbox = self.localize(pooled)
        
        return bbox, attn_map
```

#### Phase B: Visualization
Visualize the `attn_map` (Shape: `Batch, Heads, Search_Pixels, Template_Pixels`).
We want to see which pixels in the `search_img` have high attention scores with the `template` pixels.

```python
def visualize_attention(search_img, attn_map):
    # Average attention across heads and template pixels
    # attn_map: (B, Heads, N_search, N_template)
    # We want (B, N_search) -> reassemble to (H, W)
    
    attn_score = attn_map.mean(dim=(1, 3)) # Average over heads and template
    attn_score = attn_score.reshape(search_img.shape[0], search_img.shape[1], -1) 
    # Shape depends on original resolution
    
    # Normalize
    attn_score = (attn_score - attn_score.min()) / (attn_score.max() - attn_score.min())
    
    plt.imshow(search_img[0].cpu().numpy())
    plt.imshow(attn_score[0].detach().cpu().numpy(), alpha=0.6, cmap='jet')
    plt.show()
```

### 3. Expected Output
When running the tracker on two frames (one with template "ball", one search area with "ball"):
- The heat map should glow red around the "ball" in the search image.
- This demonstrates the network "attended" to the correct location based on the template features.

---

## 🚀 Project: TurtleBot4 Attention Tracking

**Goal:** Provide a ROS 2 node that subscribes to the camera, computes attention on a user-selected target, and publishes velocity commands to turn the robot towards the target.

### 1. ROS 2 Node Structure

```python
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
# import our SiameseTracker

class AttentionBot(Node):
    def __init__(self):
        super().__init__('attention_bot')
        self.sub = self.create_subscription(Image, '/camera/image_raw', self.img_cb, 10)
        self.pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.bridge = CvBridge()
        
        self.model = SiameseTracker() # Load pretrained weights
        self.template = None
        self.tracking = False
        
        # ... logic to select template via mouse click in CV window ...

    def img_cb(self, msg):
        frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        
        if self.tracking and self.template is not None:
             # Preprocess frame and template
             # Run Inference
             bbox, center_x, center_y = self.run_inference(frame)
             
             # Control Logic
             cmd = Twist()
             img_center = frame.shape[1] / 2
             error_x = center_x - img_center
             
             # P-Controller for rotation
             cmd.angular.z = -0.005 * error_x
             self.pub.publish(cmd)
```

---

## ⚡ Optimization: Flash Attention 2

For Day 2, advanced students should look into **Flash Attention 2** (Dao, 2023).

### Key concept:
- **Fewer HBM Accesses:** Fusion of Softmax operation into the matrix multiplication loop.
- **Parallelism:** Parallelize over sequence length dimension in addition to batch and heads.

**Usage in PyTorch:**
```python
# Starting PyTorch 2.1
import torch.nn.functional as F

# Automatically dispatches to FlashAttention kernel if inputs are CUDA, fp16/bf16
out = F.scaled_dot_product_attention(q, k, v, is_causal=False)
```

---

## 🧠 Assessment & Review

### Knowledge Check
1.  **Q:** What happens if you run Self-Attention on a sequence without Positional Encodings?
    *   **A:** The output is a "Bag of Words" (or patches). The model has no idea if a patch is in the top-left or bottom-right.
2.  **Q:** In Cross-Attention, why do $K$ and $V$ come from the context?
    *   **A:** The context acts as the "database" we search against. The Query is our current "search term". We want to retrieve values from the context that match our search term.
3.  **Q:** Why is Flash Attention faster if it does the same math?
    *   **A:** It is memory-bound, not compute-bound. Flash Attention reduces the number of read/write operations to the slow GPU Global Memory (HBM).

### Challenge Task
> **Task:** Modify the `SiameseTracker` to use Multi-Scale features.
> 1. Extract features from a CNN at 3 different layers.
> 2. Perform Cross-Attention at each scale.
> 3. Upsample and Fuse the attention maps.

---

## 📚 Further Reading
- **Attention Is All You Need** (Vaswani et al., NeurlPS 2017)
- **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** (Dao et al., ArXiv 2022)
- **Cross-View Transformer for Real-Time Surround Perception** (Zhou et al.)

---

**Day 2 Complete**

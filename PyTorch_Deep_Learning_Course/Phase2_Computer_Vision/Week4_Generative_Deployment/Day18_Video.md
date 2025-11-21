# Day 18: Video Understanding - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: 3D CNNs, Optical Flow, and Video Transformers

## 1. Theoretical Foundation: The Time Dimension

Video = Images + Time ($T \times C \times H \times W$).
Challenges:
1.  **Data Volume**: Videos are huge.
2.  **Temporal Redundancy**: Adjacent frames are nearly identical.
3.  **Motion**: Understanding actions requires tracking movement, not just appearance.

### Approaches
1.  **2D CNN + RNN**: Extract features per frame, aggregate with LSTM. (Old school).
2.  **3D CNN (C3D, I3D)**: Convolve over Space and Time ($K_t \times K_h \times K_w$).
3.  **Two-Stream**: Spatial Stream (RGB) + Temporal Stream (Optical Flow).
4.  **Video Transformers**: ViViT, TimeSformer.

## 2. 3D Convolutions (C3D)

Standard Conv2d: $(C, H, W) \to (C, H, W)$.
Conv3d: $(C, T, H, W) \to (C, T, H, W)$.
*   Kernel: $3 \times 3 \times 3$.
*   Captures spatiotemporal features (e.g., "Arm moving up").

## 3. Implementation: R(2+1)D

Full 3D convolution is expensive ($K^3$ params).
**Factorized 3D Conv**:
Decompose $3 \times 3 \times 3$ into:
1.  **Spatial Conv**: $1 \times 3 \times 3$.
2.  **Temporal Conv**: $3 \times 1 \times 1$.
*   Similar to Depthwise Separable, but for Time/Space.
*   More non-linearities, fewer parameters, easier to optimize.

```python
import torch
import torch.nn as nn
from torchvision.models.video import r2plus1d_18

# Load Pre-trained R(2+1)D
model = r2plus1d_18(weights='DEFAULT')
model.eval()

# Input: (Batch, Channel, Time, Height, Width)
# 8 frames, 112x112
x = torch.randn(1, 3, 8, 112, 112)
out = model(x) # Class logits
```

## 4. Video Transformers (TimeSformer)

Applying Self-Attention to video.
Naive approach: Flatten all patches in all frames. $O((THW)^2)$ complexity. Impossible.
**Divided Space-Time Attention**:
1.  **Temporal Attention**: Compare patch $(h, w)$ across all frames $t$.
2.  **Spatial Attention**: Compare all patches in frame $t$.
*   Linear complexity $O(T + HW)$.

## 5. Data Loading for Video

Handling video is tricky.
*   **Sampling**: Don't take every frame. Sample 8 frames uniformly from the clip.
*   **Decoding**: Decoding MP4 on the fly is CPU intensive. Use `torchvision.io.read_video` or NVIDIA DALI.

```python
# Sampling Strategy
def sample_frames(video_path, num_frames=8):
    # Read video
    vframes, _, _ = torchvision.io.read_video(video_path)
    total_frames = len(vframes)
    # Uniform indices
    indices = torch.linspace(0, total_frames-1, num_frames).long()
    return vframes[indices]
```

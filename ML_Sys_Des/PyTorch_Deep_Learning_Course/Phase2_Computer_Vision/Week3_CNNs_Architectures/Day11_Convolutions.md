# Day 11: Convolutions - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: The Convolution Operation, Kernels, and Receptive Fields

## 1. Theoretical Foundation: Why Convolutions?

Fully Connected (Dense) layers fail for images because:
1.  **Parameter Explosion**: A 1000x1000 image has 1M pixels. A dense layer with 1000 hidden units would have $10^9$ weights.
2.  **Spatial Invariance**: A cat in the top-left corner is the same as a cat in the bottom-right. Dense layers don't share weights.

### The Convolution Operation
$$ (I * K)(i, j) = \sum_m \sum_n I(i+m, j+n) K(m, n) $$
*   **Sparse Connectivity**: Each output depends on a small region (kernel size).
*   **Parameter Sharing**: The same kernel $K$ is used everywhere.
*   **Translation Equivariance**: Shift input $\to$ Shift output.

## 2. Key Concepts

### Kernel Size ($K$)
*   Small ($3 \times 3$): Captures fine details. Most common (VGG style).
*   Large ($7 \times 7$): Captures global context early. Used in first layer of ResNet/ViT.

### Stride ($S$)
*   Step size of the sliding window.
*   $S=1$: Output size $\approx$ Input size.
*   $S=2$: Downsamples by factor of 2 (Alternative to Pooling).

### Padding ($P$)
*   Adding zeros around the border.
*   **Valid**: No padding. Output shrinks.
*   **Same**: Pad such that Output size = Input size (if $S=1$).
*   Formula: $H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} + 1 \rfloor$

### Channels ($C$)
*   Input: $(N, C_{in}, H, W)$.
*   Output: $(N, C_{out}, H_{out}, W_{out})$.
*   The kernel actually has shape $(C_{out}, C_{in}, K, K)$. It is 3D!

## 3. PyTorch Implementation

```python
import torch
import torch.nn as nn

# 1. Standard Conv2d
conv = nn.Conv2d(
    in_channels=3,    # RGB
    out_channels=64,  # Number of filters
    kernel_size=3,
    stride=1,
    padding=1         # "Same" padding for K=3
)

# 2. Pooling (Downsampling)
pool = nn.MaxPool2d(kernel_size=2, stride=2)

# 3. A Simple CNN Block
class ConvBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn = nn.BatchNorm2d(out_c)
        self.relu = nn.ReLU()
        
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
```

## 4. Receptive Field (RF)

The region of the input image that a particular neuron "sees".
*   Layer 1 ($3 \times 3$): RF = 3.
*   Layer 2 ($3 \times 3$ on top of Layer 1): RF = 5.
*   Layer 3: RF = 7.
*   RF increases linearly with depth.
*   **Goal**: The final layer's RF should cover the entire object.

## 5. Dilated Convolution (Atrous)

How to increase RF exponentially without losing resolution (pooling)?
Expand the kernel by inserting holes (dilation rate $d$).
*   $K_{eff} = K + (K-1)(d-1)$.
*   Used in Segmentation (DeepLab) and Audio (WaveNet).

```python
# Receptive field of 5x5 with only 3x3 parameters
dilated_conv = nn.Conv2d(64, 64, 3, padding=2, dilation=2)
```

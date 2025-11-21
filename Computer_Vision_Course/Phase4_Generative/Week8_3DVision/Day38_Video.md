# Day 38: Video Understanding

## 1. The Temporal Dimension
Video = Space ($H \times W$) + Time ($T$).
Input shape: $(B, C, T, H, W)$.
**Challenges:**
*   Huge data volume.
*   Temporal redundancy (adjacent frames are similar).
*   Motion modeling is complex.

## 2. 3D CNNs (C3D, I3D)
**Idea:** Extend 2D convolution to 3D.
*   Kernel: $k_t \times k_h \times k_w$ (e.g., $3 \times 3 \times 3$).
*   Moves through Time, Height, and Width.
*   **C3D (2015):** VGG-like architecture with 3D convs.
*   **I3D (Inception-3D):** Inflated Inception-v1.
    *   Initialize 3D weights by replicating 2D ImageNet weights along time.
    *   State-of-the-art for a long time.

## 3. Two-Stream Networks
**Idea:** Separate Appearance and Motion.
1.  **Spatial Stream:** Input is single frame (RGB). Captures "What" (e.g., Tennis racket).
2.  **Temporal Stream:** Input is Optical Flow stack. Captures "How" (e.g., Swinging motion).
3.  **Fusion:** Average predictions at the end.

## 4. SlowFast Networks (2019)
**Idea:** Inspired by biological vision (Parvocellular vs Magnocellular pathways).
1.  **Slow Pathway:**
    *   Low frame rate ($T/16$).
    *   High channel capacity.
    *   Captures **Spatial Semantics**.
2.  **Fast Pathway:**
    *   High frame rate ($T/2$).
    *   Low channel capacity ($\beta C$).
    *   Captures **Motion**.
*   **Lateral Connections:** Fuse Fast features into Slow pathway.

```python
import torch
import torch.nn as nn

class SlowFastBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Fast Path (High T, Low C)
        self.fast_conv = nn.Conv3d(in_channels//8, out_channels//8, kernel_size=(5, 1, 1), padding=(2, 0, 0))
        # Slow Path (Low T, High C)
        self.slow_conv = nn.Conv3d(in_channels, out_channels, kernel_size=(1, 1, 1))
        # Lateral Connection
        self.lateral = nn.Conv3d(out_channels//8, out_channels, kernel_size=(5, 1, 1), padding=(2, 0, 0), stride=(8, 1, 1))

    def forward(self, slow_input, fast_input):
        fast_out = self.fast_conv(fast_input)
        slow_out = self.slow_conv(slow_input)
        # Fuse
        slow_out = slow_out + self.lateral(fast_out)
        return slow_out, fast_out
```

## Summary
Video understanding moved from simple 2D CNNs + LSTM to specialized 3D architectures like SlowFast that explicitly model the trade-off between spatial and temporal resolution.

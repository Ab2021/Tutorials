# Day 27: DeepLab & Dilated Convolutions

## 1. The Problem with Pooling
Standard CNNs use Max Pooling to increase receptive field.
*   **Pro:** Captures global context.
*   **Con:** Reduces spatial resolution (loss of detail).
*   **Result:** Up-sampling from $1/32$ scale produces blurry segmentation masks.

## 2. Atrous (Dilated) Convolution
**Idea:** Expand the kernel by inserting holes (zeros) between weights.
*   **Rate ($r$):** The spacing between weights.
*   **Receptive Field:** A $3 \times 3$ kernel with rate $r=2$ has the same receptive field as a $5 \times 5$ kernel, but with same parameters (9).
*   **Benefit:** Increases receptive field **without** downsampling resolution. We can keep the feature map at $1/8$ scale instead of $1/32$.

## 3. DeepLab Evolution

### DeepLab v1 (2015)
*   Used Atrous Conv to keep resolution high.
*   **CRF (Conditional Random Field):** Post-processing step to refine boundaries.

### DeepLab v2 (2016)
*   **ASPP (Atrous Spatial Pyramid Pooling):**
    *   Apply multiple atrous convs with different rates ($r=6, 12, 18, 24$) in parallel.
    *   Captures objects at multiple scales.

### DeepLab v3 (2017)
*   Improved ASPP with Image-level features (Global Avg Pool).
*   Removed CRF (network is good enough).

### DeepLab v3+ (2018)
*   Added a simple **Decoder** module to refine object boundaries (like U-Net).
*   Used **Xception** backbone with Depthwise Separable Convolutions for speed.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ASPP, self).__init__()
        # 1x1 Conv
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        # Atrous 3x3, rate=6
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, padding=6, dilation=6, bias=False)
        # Atrous 3x3, rate=12
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, padding=12, dilation=12, bias=False)
        # Atrous 3x3, rate=18
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, padding=18, dilation=18, bias=False)
        # Global Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.conv5 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        
        self.bn = nn.BatchNorm2d(out_channels * 5)
        self.relu = nn.ReLU()
        self.out_conv = nn.Conv2d(out_channels * 5, out_channels, 1, bias=False)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        x4 = self.conv4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(self.conv5(x5), size=x.shape[2:], mode='bilinear', align_corners=True)
        
        x = torch.cat([x1, x2, x3, x4, x5], dim=1)
        return self.out_conv(self.relu(self.bn(x)))
```

## Summary
DeepLab introduced Atrous Convolution to solve the resolution vs. context trade-off. ASPP allows the model to "zoom in" and "zoom out" simultaneously to handle multi-scale objects.

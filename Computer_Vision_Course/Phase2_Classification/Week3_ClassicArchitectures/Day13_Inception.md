# Day 13: Inception & Network in Network

## 1. Network in Network (NiN, 2013)
**Key Innovation:** $1 \times 1$ Convolution.
*   **Problem:** Linear filters in CNNs are not expressive enough.
*   **Solution:** Use a micro-network (MLP) inside the filter.
*   **Implementation:** This is mathematically equivalent to a $1 \times 1$ convolution followed by ReLU.

**Benefits of $1 \times 1$ Conv:**
1.  **Dimensionality Reduction:** Reduce channels (e.g., $256 \to 64$) to save computation.
2.  **Increased Non-linearity:** Adds depth and activation without changing spatial size.
3.  **Cross-Channel Information:** Mixes information across channels at a single pixel.

## 2. Inception (GoogLeNet, 2014)
**Philosophy:** Go Wider, not just Deeper.
*   **Problem:** Choosing the right kernel size ($1 \times 1, 3 \times 3, 5 \times 5$) is hard. Small kernels capture local details, large kernels capture global context.
*   **Solution:** Use **all of them** in parallel and concatenate the results.

### Naive Inception Module
*   Parallel paths: $1 \times 1$, $3 \times 3$, $5 \times 5$, MaxPool.
*   **Issue:** Computational explosion. $5 \times 5$ on large channels is expensive.

### Inception Module with Dimension Reduction
*   Use $1 \times 1$ convs **before** expensive $3 \times 3$ and $5 \times 5$ convs to reduce channels.
*   **Bottleneck Design:** Compress $\to$ Process $\to$ Concatenate.

**Architecture (GoogLeNet):**
*   22 layers deep.
*   9 Inception modules.
*   **Auxiliary Classifiers:** Small heads attached to middle layers to inject gradients and combat vanishing gradient (deprecated in later versions).
*   **Global Average Pooling:** Replaced FC layers.

```python
import torch
import torch.nn as nn

class InceptionModule(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModule, self).__init__()
        
        # 1x1 branch
        self.branch1 = BasicConv2d(in_channels, ch1x1, kernel_size=1)
        
        # 3x3 branch
        self.branch2 = nn.Sequential(
            BasicConv2d(in_channels, ch3x3red, kernel_size=1),
            BasicConv2d(ch3x3red, ch3x3, kernel_size=3, padding=1)
        )
        
        # 5x5 branch
        self.branch3 = nn.Sequential(
            BasicConv2d(in_channels, ch5x5red, kernel_size=1),
            BasicConv2d(ch5x5red, ch5x5, kernel_size=5, padding=2)
        )
        
        # Pool branch
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            BasicConv2d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        
        return torch.cat([branch1, branch2, branch3, branch4], 1)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return self.relu(x)
```

## 3. Xception (Extreme Inception, 2016)
**Hypothesis:** Cross-channel correlations and spatial correlations can be mapped completely separately.
*   **Depthwise Separable Convolution:**
    1.  **Depthwise:** $3 \times 3$ conv on each channel separately (Spatial).
    2.  **Pointwise:** $1 \times 1$ conv across all channels (Cross-channel).
*   **Result:** Same accuracy as Inception V3 but fewer parameters and faster.

## Summary
Inception introduced the idea of multi-scale processing and efficient bottleneck designs using $1 \times 1$ convolutions. Xception pushed this to the limit with Depthwise Separable Convolutions, a key component of modern efficient networks (MobileNets).

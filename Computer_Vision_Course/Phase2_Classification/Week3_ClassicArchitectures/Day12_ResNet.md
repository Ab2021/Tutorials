# Day 12: ResNet & Skip Connections

## 1. The Vanishing Gradient Problem
As networks get deeper (e.g., 20+ layers), accuracy gets **worse**, not better.
*   **Not overfitting:** Training error also increases.
*   **Cause:** Gradients vanish (multiply by small numbers) or explode (multiply by large numbers) as they backpropagate through many layers.
*   **Result:** Early layers stop learning.

## 2. Residual Learning (ResNet, 2015)
**Key Insight:** It's harder to learn a mapping $H(x)$ than to learn a residual mapping $F(x) = H(x) - x$.
*   **Hypothesis:** If identity mapping is optimal, it's easier to push residuals to zero ($F(x) \to 0$) than to fit an identity function ($H(x) \to x$) with non-linear layers.

**Skip Connection (Shortcut):**
$$ y = F(x, \{W_i\}) + x $$
*   $x$: Input to the block.
*   $F(x)$: Residual function (e.g., Conv-ReLU-Conv).
*   $+$: Element-wise addition.

**Benefit:**
*   Gradient can flow directly through the skip connection ($+x$) during backprop.
*   Acts as a "gradient superhighway."
*   Allows training of very deep networks (100+ layers).

## 3. ResNet Architectures

### Basic Block (ResNet-18, 34)
*   Two $3 \times 3$ convolutions.
*   Used for shallower networks.

### Bottleneck Block (ResNet-50, 101, 152)
Used to reduce parameters and computation.
1.  **$1 \times 1$ Conv:** Reduce dimensions (e.g., $256 \to 64$).
2.  **$3 \times 3$ Conv:** Process features ($64 \to 64$).
3.  **$1 \times 1$ Conv:** Restore dimensions ($64 \to 256$).

**Structure (ResNet-50):**
1.  **Stem:** $7 \times 7$ Conv, MaxPool.
2.  **Stage 1:** 3 Bottleneck blocks (64 filters).
3.  **Stage 2:** 4 Bottleneck blocks (128 filters).
4.  **Stage 3:** 6 Bottleneck blocks (256 filters).
5.  **Stage 4:** 3 Bottleneck blocks (512 filters).
6.  **Head:** Global Avg Pool, FC-1000.

```python
import torch.nn as nn

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 1x1 conv (Reduce)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        
        # 3x3 conv (Process)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        
        # 1x1 conv (Restore)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
```

## 4. ResNeXt (2017)
**Idea:** Split the bottleneck into multiple parallel paths (Cardinality).
*   Like Inception, but paths are identical.
*   **Grouped Convolution:** Efficient implementation.
*   Better accuracy than ResNet for same parameter count.

## Summary
ResNet solved the depth problem, enabling 100+ layer networks. It is the default backbone for almost all computer vision tasks today.

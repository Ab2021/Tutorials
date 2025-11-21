# Day 12: Modern Backbones - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: ResNet, EfficientNet, and ConvNeXt

## 1. Theoretical Foundation: The Vanishing Gradient Problem

As networks get deeper, gradients vanish (multiply by small numbers many times) or explode.
VGG (19 layers) was the limit. ResNet (152 layers) broke this barrier.

### ResNet: The Residual Connection
Hypothesis: It is easier to learn the residual mapping $F(x) = H(x) - x$ than the original mapping $H(x)$.
$$ y = F(x) + x $$
*   **Identity Shortcut**: Gradients can flow directly through the $+x$ path to earlier layers without attenuation.
*   **Result**: We can train networks with 1000+ layers.

## 2. Evolution of Architectures

### VGG (2014)
*   Stack of $3 \times 3$ convs.
*   Heavy, slow, lots of parameters (138M).

### ResNet (2015)
*   Skip connections.
*   **Bottleneck Block**: $1\times1 \to 3\times3 \to 1\times1$.
*   Standard backbone for years.

### EfficientNet (2019)
*   **Compound Scaling**: Optimally balancing Depth, Width, and Resolution.
*   Uses **MBConv** (Mobile Inverted Bottleneck) from MobileNetV2.
*   Squeeze-and-Excitation (SE) blocks.

### ConvNeXt (2022)
*   "A ConvNet for the 2020s".
*   Modernizing ResNet to look like a Transformer (Swin).
*   Large kernels ($7 \times 7$), LayerNorm, GELU, fewer activations.
*   Outperforms Swin Transformer on ImageNet.

## 3. Implementation: ResNet Block

```python
import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
            
    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The Magic
        out = torch.relu(out)
        return out
```

## 4. Using Pre-trained Models (TorchVision)

Don't reinvent the wheel. Use `torchvision.models`.

```python
from torchvision import models

# 1. Load Architecture
resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)

# 2. Inspect
print(resnet)

# 3. Feature Extraction (Remove last layer)
backbone = nn.Sequential(*list(resnet.children())[:-1])
features = backbone(torch.randn(1, 3, 224, 224)) # (1, 2048, 1, 1)
```

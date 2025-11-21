# Day 8 Deep Dive: Architecture Design Principles

## 1. Network Design Patterns

### Pattern 1: Increasing Depth
**Historical progression:**
- AlexNet (2012): 8 layers
- VGG (2014): 16-19 layers
- ResNet (2015): 50-152 layers
- ResNet-1000+ (2016): 1000+ layers

**Why deeper is better:**
- More hierarchical features
- Larger receptive fields
- Better abstraction

**Challenge:** Degradation problem (solved by ResNet).

### Pattern 2: Decreasing Spatial Dimensions
**Typical pattern:**
```
224×224 → 112×112 → 56×56 → 28×28 → 14×14 → 7×7 → 1×1
```

**Methods:**
1. **Strided convolution:** $s > 1$
2. **Pooling:** Max or average
3. **Dilated convolution:** Increase receptive field without downsampling

### Pattern 3: Increasing Channel Depth
**Typical progression:**
```
3 → 64 → 128 → 256 → 512 → 1024 → 2048
```

**Rationale:** As spatial dimensions decrease, increase feature complexity.

## 2. Advanced ResNet Variants

### ResNeXt (2017)
**Idea:** "Cardinality" (number of paths) as new dimension.

**Split-transform-merge:**
```python
class ResNeXtBlock(nn.Module):
    """ResNeXt block with grouped convolutions."""
    
    def __init__(self, in_channels, out_channels, cardinality=32, base_width=4, stride=1):
        super(ResNeXtBlock, self).__init__()
        
        width = int(out_channels * (base_width / 64.)) * cardinality
        
        self.conv1 = nn.Conv2d(in_channels, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        
        # Grouped convolution
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                              padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        
        self.conv3 = nn.Conv2d(width, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

**Complexity:**
$$ C \approx \frac{256 \cdot d}{64} \cdot 9 \cdot C_{card} $$

where $C_{card}$ is cardinality.

### SE-ResNet (Squeeze-and-Excitation)
**Idea:** Channel-wise attention mechanism.

**SE Block:**
1. **Squeeze:** Global average pooling → $1 \times 1 \times C$
2. **Excitation:** FC → ReLU → FC → Sigmoid → channel weights
3. **Scale:** Multiply input by channel weights

```python
class SEBlock(nn.Module):
    """Squeeze-and-Excitation block."""
    
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        batch, channels, _, _ = x.size()
        
        # Squeeze
        y = self.squeeze(x).view(batch, channels)
        
        # Excitation
        y = self.excitation(y).view(batch, channels, 1, 1)
        
        # Scale
        return x * y.expand_as(x)

class SEResNetBlock(nn.Module):
    """ResNet block with SE."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(SEResNetBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * 4)
        
        self.se = SEBlock(out_channels * 4)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        
        if stride != 1 or in_channels != out_channels * 4:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * 4),
            )
    
    def forward(self, x):
        identity = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        
        out = self.se(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out
```

## 3. MobileNet Family

### MobileNet V1 (2017)
**Key:** Depthwise separable convolutions.

**Standard conv FLOPs:** $H \times W \times C_{in} \times C_{out} \times k^2$
**Depthwise separable FLOPs:** $H \times W \times (C_{in} \times k^2 + C_{in} \times C_{out})$

**Reduction:** $\frac{1}{C_{out}} + \frac{1}{k^2}$

### MobileNet V2 (2018)
**Inverted residual block:**
1. Expand with 1×1 conv (expansion factor $t=6$)
2. Depthwise 3×3 conv
3. Project with 1×1 conv (no ReLU!)

**Linear bottleneck:** Last layer has no activation (preserves information).

```python
class InvertedResidual(nn.Module):
    """MobileNetV2 inverted residual block."""
    
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        layers = []
        
        # Expand
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
            ])
        
        # Depthwise
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride,
                     padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
        ])
        
        # Project (linear bottleneck - no activation!)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_residual:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

### MobileNet V3 (2019)
**Improvements:**
1. **Neural Architecture Search (NAS):** Automated design
2. **h-swish activation:** $x \cdot \frac{\text{ReLU6}(x + 3)}{6}$
3. **SE blocks:** Channel attention
4. **Redesigned expensive layers:** Reduce last layers' cost

## 4. EfficientNet Deep Dive

### Compound Scaling Method
**Baseline (EfficientNet-B0):** Found via NAS.

**Scaling:**
$$ \text{depth: } d = \alpha^\phi $$
$$ \text{width: } w = \beta^\phi $$
$$ \text{resolution: } r = \gamma^\phi $$

**Constraint:** $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

**Grid search results:** $\alpha = 1.2, \beta = 1.1, \gamma = 1.15$

**FLOPs scaling:** Approximately $2^\phi$

### MBConv Block
**Mobile inverted bottleneck convolution:**
```python
class MBConvBlock(nn.Module):
    """EfficientNet MBConv block."""
    
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        
        hidden_dim = in_channels * expand_ratio
        self.use_residual = stride == 1 and in_channels == out_channels
        
        # Expansion
        if expand_ratio != 1:
            self.expand_conv = nn.Sequential(
                nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.SiLU(inplace=True),
            )
        else:
            self.expand_conv = nn.Identity()
        
        # Depthwise
        self.depthwise_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size,
                     stride=stride, padding=kernel_size//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.SiLU(inplace=True),
        )
        
        # Squeeze-and-Excitation
        if se_ratio > 0:
            se_channels = max(1, int(in_channels * se_ratio))
            self.se = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(hidden_dim, se_channels, kernel_size=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(se_channels, hidden_dim, kernel_size=1),
                nn.Sigmoid(),
            )
        else:
            self.se = None
        
        # Projection
        self.project_conv = nn.Sequential(
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        identity = x
        
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        
        if self.se is not None:
            x = x * self.se(x)
        
        x = self.project_conv(x)
        
        if self.use_residual:
            x = x + identity
        
        return x
```

## 5. Architecture Comparison

| Model | Year | Params | FLOPs | Top-1 Acc | Key Innovation |
|-------|------|--------|-------|-----------|----------------|
| AlexNet | 2012 | 61M | 720M | 63.3% | ReLU, Dropout, GPU |
| VGG-16 | 2014 | 138M | 15.5B | 71.5% | Deep + small filters |
| ResNet-50 | 2015 | 25.6M | 4.1B | 76.2% | Residual connections |
| ResNet-152 | 2015 | 60.2M | 11.6B | 77.6% | Very deep |
| DenseNet-201 | 2017 | 20M | 4.3B | 77.4% | Dense connections |
| MobileNetV2 | 2018 | 3.5M | 300M | 72.0% | Inverted residuals |
| EfficientNet-B0 | 2019 | 5.3M | 390M | 77.1% | Compound scaling |
| EfficientNet-B7 | 2019 | 66M | 37B | 84.4% | Scaled B0 |

## Summary
Modern architectures balance accuracy, efficiency, and design principles. Key innovations include residual connections, efficient convolutions, attention mechanisms, and automated architecture search.

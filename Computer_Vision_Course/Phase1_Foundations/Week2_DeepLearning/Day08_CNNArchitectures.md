# Day 8: CNN Architectures

## Overview
This lesson explores landmark CNN architectures that revolutionized computer vision: AlexNet, VGGNet, ResNet, DenseNet, and EfficientNet. We'll understand their design principles, innovations, and implementation.

## 1. AlexNet (2012)

### Architecture
**ImageNet winner 2012, Top-5 error: 15.3%**

```
Input (227×227×3)
↓
Conv1: 96 filters, 11×11, stride 4 → ReLU → MaxPool (3×3, stride 2)
↓
Conv2: 256 filters, 5×5, stride 1 → ReLU → MaxPool (3×3, stride 2)
↓
Conv3: 384 filters, 3×3, stride 1 → ReLU
↓
Conv4: 384 filters, 3×3, stride 1 → ReLU
↓
Conv5: 256 filters, 3×3, stride 1 → ReLU → MaxPool (3×3, stride 2)
↓
FC6: 4096 → ReLU → Dropout(0.5)
↓
FC7: 4096 → ReLU → Dropout(0.5)
↓
FC8: 1000 (softmax)
```

**Key Innovations:**
1. **ReLU activation:** First large-scale use, 6× faster than tanh
2. **Dropout:** Regularization in FC layers
3. **Data augmentation:** Random crops, horizontal flips, color jittering
4. **GPU training:** Parallelized across 2 GPUs
5. **Local Response Normalization (LRN):** Lateral inhibition

### Implementation

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## 2. VGGNet (2014)

### Architecture
**Key insight:** Deeper networks with smaller filters (3×3) are more effective.

**VGG-16:**
```
Input (224×224×3)
↓
[Conv3-64] × 2 → MaxPool
↓
[Conv3-128] × 2 → MaxPool
↓
[Conv3-256] × 3 → MaxPool
↓
[Conv3-512] × 3 → MaxPool
↓
[Conv3-512] × 3 → MaxPool
↓
FC-4096 → FC-4096 → FC-1000
```

**Design Principles:**
1. **Small receptive fields:** 3×3 convolutions only
2. **Deep architecture:** 16-19 layers
3. **Uniform design:** Same pattern repeated
4. **1×1 convolutions:** For dimensionality manipulation

**Receptive field equivalence:**
- Two 3×3 convs = one 5×5 conv (but fewer parameters)
- Three 3×3 convs = one 7×7 conv

**Parameters:**
- VGG-16: 138M parameters (mostly in FC layers)
- VGG-19: 144M parameters

### Implementation

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

## 3. ResNet (2015)

### The Degradation Problem
**Observation:** Deeper networks have higher training error (not just overfitting).

**Hypothesis:** Deep networks struggle to learn identity mappings.

### Residual Learning
**Key idea:** Learn residual function $F(x) = H(x) - x$ instead of $H(x)$.

**Residual block:**
$$ y = F(x, \{W_i\}) + x $$

where $F(x, \{W_i\})$ represents stacked layers.

**Gradient flow:**
$$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \left(1 + \frac{\partial F}{\partial x}\right) $$

The "+1" ensures gradient always flows, preventing vanishing gradients.

### Architecture

**ResNet-50:**
```
Input (224×224×3)
↓
Conv1: 7×7, 64, stride 2
↓
MaxPool: 3×3, stride 2
↓
Stage 1: [1×1,64; 3×3,64; 1×1,256] × 3
↓
Stage 2: [1×1,128; 3×3,128; 1×1,512] × 4
↓
Stage 3: [1×1,256; 3×3,256; 1×1,1024] × 6
↓
Stage 4: [1×1,512; 3×3,512; 1×1,2048] × 3
↓
Global Average Pooling
↓
FC-1000
```

**Bottleneck block:**
- 1×1 conv: Reduce dimensions (256 → 64)
- 3×3 conv: Process features (64 → 64)
- 1×1 conv: Restore dimensions (64 → 256)

### Implementation

```python
class ResidualBlock(nn.Module):
    """Bottleneck residual block."""
    expansion = 4
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        
        # Bottleneck
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, 
                              kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
    
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

class ResNet50(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet50, self).__init__()
        
        self.in_channels = 64
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * ResidualBlock.expansion, num_classes)
    
    def _make_layer(self, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * ResidualBlock.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * ResidualBlock.expansion,
                         kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * ResidualBlock.expansion),
            )
        
        layers = []
        layers.append(ResidualBlock(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * ResidualBlock.expansion
        
        for _ in range(1, blocks):
            layers.append(ResidualBlock(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
```

## 4. DenseNet (2017)

### Dense Connectivity
**Idea:** Connect each layer to every subsequent layer.

**Layer $l$ receives:**
$$ x_l = H_l([x_0, x_1, ..., x_{l-1}]) $$

where $[x_0, x_1, ..., x_{l-1}]$ is concatenation.

**Advantages:**
1. **Feature reuse:** Earlier features accessible to all layers
2. **Gradient flow:** Direct paths to all layers
3. **Regularization:** Implicit deep supervision
4. **Parameter efficiency:** Fewer parameters than ResNet

**Growth rate $k$:** Number of feature maps each layer adds.

### Implementation

```python
class DenseBlock(nn.Module):
    def __init__(self, num_layers, in_channels, growth_rate):
        super(DenseBlock, self).__init__()
        self.layers = nn.ModuleList()
        
        for i in range(num_layers):
            self.layers.append(
                self._make_layer(in_channels + i * growth_rate, growth_rate)
            )
    
    def _make_layer(self, in_channels, growth_rate):
        return nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False),
            nn.BatchNorm2d(4 * growth_rate),
            nn.ReLU(inplace=True),
            nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False),
        )
    
    def forward(self, x):
        features = [x]
        for layer in self.layers:
            new_features = layer(torch.cat(features, 1))
            features.append(new_features)
        return torch.cat(features, 1)
```

## 5. EfficientNet (2019)

### Compound Scaling
**Insight:** Balance depth, width, and resolution.

**Scaling:**
$$ \text{depth: } d = \alpha^\phi $$
$$ \text{width: } w = \beta^\phi $$
$$ \text{resolution: } r = \gamma^\phi $$

subject to $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ and $\alpha, \beta, \gamma \geq 1$.

**EfficientNet-B0 to B7:** Scale with $\phi = 0, 1, ..., 7$.

## Summary
- **AlexNet:** Pioneered deep CNNs
- **VGG:** Demonstrated depth with small filters
- **ResNet:** Solved degradation with residual connections
- **DenseNet:** Maximized feature reuse
- **EfficientNet:** Optimized scaling

**Next:** Training techniques and best practices.

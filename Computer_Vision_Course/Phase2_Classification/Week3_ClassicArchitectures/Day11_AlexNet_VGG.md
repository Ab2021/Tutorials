# Day 11: AlexNet & VGGNet - The Deep Learning Revolution

## 1. AlexNet (2012)
**Significance:** The model that started the Deep Learning revolution by winning the ImageNet Large Scale Visual Recognition Challenge (ILSVRC) 2012 by a huge margin (Top-5 error 15.3% vs 26.2% for runner-up).

**Key Innovations:**
1.  **ReLU Nonlinearity:** Faster training than Tanh/Sigmoid.
2.  **Multi-GPU Training:** Split model across 2 GPUs (historical constraint).
3.  **Local Response Normalization (LRN):** Later replaced by Batch Norm.
4.  **Overlapping Pooling:** Max pooling with stride < kernel size.
5.  **Dropout:** Used in FC layers to prevent overfitting.

**Architecture:**
*   Input: $227 \times 227 \times 3$
*   5 Convolutional Layers
*   3 Fully Connected Layers
*   Output: 1000 classes (Softmax)

```python
import torch
import torch.nn as nn

class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # Conv 1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv 2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # Conv 3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv 4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv 5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
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
**Philosophy:** Simplicity and Depth.
*   Instead of large filters ($11 \times 11, 5 \times 5$), use stack of small $3 \times 3$ filters.
*   **Key Insight:** Two $3 \times 3$ convs have same receptive field as one $5 \times 5$ but fewer parameters and more non-linearity.

**Architecture (VGG-16):**
*   **Blocks:** Conv-Conv-Pool sequence repeated.
*   **Filters:** Doubles after each pooling (64 $\to$ 128 $\to$ 256 $\to$ 512).
*   **Uniformity:** All convs are $3 \times 3$, stride 1, padding 1. All pools are $2 \times 2$, stride 2.

**VGG-16 Configuration:**
1.  Conv3-64, Conv3-64, MaxPool
2.  Conv3-128, Conv3-128, MaxPool
3.  Conv3-256, Conv3-256, Conv3-256, MaxPool
4.  Conv3-512, Conv3-512, Conv3-512, MaxPool
5.  Conv3-512, Conv3-512, Conv3-512, MaxPool
6.  FC-4096, FC-4096, FC-1000

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()
        self.features = self._make_layers([64, 64, 'M', 128, 128, 'M', 
                                           256, 256, 256, 'M', 
                                           512, 512, 512, 'M', 
                                           512, 512, 512, 'M'])
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=3, padding=1),
                           nn.BatchNorm2d(v), # Modern addition
                           nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)
```

## 3. Comparison
| Feature | AlexNet | VGG-16 |
| :--- | :--- | :--- |
| **Year** | 2012 | 2014 |
| **Layers** | 8 | 16 |
| **Filter Size** | 11, 5, 3 | 3 only |
| **Parameters** | ~60M | ~138M |
| **Top-5 Error** | 15.3% | 7.3% |

## Summary
AlexNet proved CNNs work. VGGNet showed that depth and small uniform filters are the way forward, establishing the standard design pattern for years.

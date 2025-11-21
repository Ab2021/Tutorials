# Day 15: Semantic & Instance Segmentation - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: U-Net, Mask R-CNN, and Pixel-wise Classification

## 1. Theoretical Foundation: Types of Segmentation

1.  **Semantic Segmentation**: Classify every pixel. (All cars are "Car").
2.  **Instance Segmentation**: Detect and Segment. (Car #1, Car #2).
3.  **Panoptic Segmentation**: Combine both (Stuff + Things).

### The Challenge
We need output resolution equal to input resolution ($H \times W$).
But CNNs downsample ($H/32 \times W/32$).
We need to **Upsample** back to original size while recovering spatial details.

## 2. U-Net Architecture

Originally for Biomedical Imaging (2015). The standard for segmentation.
**Encoder-Decoder** with **Skip Connections**.
*   **Encoder**: ResNet/VGG. Downsamples, extracts semantics.
*   **Decoder**: Upsamples.
*   **Skip Connections**: Concatenate high-res features from Encoder with upsampled features from Decoder. Recovers fine details (edges).

## 3. Mask R-CNN (Instance Segmentation)

Extends Faster R-CNN.
*   Adds a third branch: **Mask Branch**.
*   Predicts a binary mask ($28 \times 28$) for each RoI.
*   Uses **RoI Align** (precise) instead of RoI Pooling (quantized).

## 4. Implementation: Simple U-Net Block

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(),
            nn.Conv2d(out_c, out_c, 3, 1, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    def forward(self, x): return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_c=3, out_c=1):
        super().__init__()
        self.down1 = DoubleConv(in_c, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.down2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        
        self.bottleneck = DoubleConv(128, 256)
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.up_conv1 = DoubleConv(256, 128) # 256 due to concat
        
        self.out = nn.Conv2d(128, out_c, 1)
        
    def forward(self, x):
        x1 = self.down1(x)
        p1 = self.pool1(x1)
        x2 = self.down2(p1)
        p2 = self.pool2(x2)
        
        b = self.bottleneck(p2)
        
        u1 = self.up1(b)
        # Concatenate Skip Connection
        u1 = torch.cat((x2, u1), dim=1) 
        x = self.up_conv1(u1)
        
        return self.out(x)
```

## 5. Loss Functions

Cross Entropy works, but **Dice Loss** is better for class imbalance (small foreground).
$$ Dice = \frac{2 |A \cap B|}{|A| + |B|} $$
```python
def dice_loss(pred, target, smooth=1.):
    pred = pred.sigmoid()
    intersection = (pred * target).sum()
    return 1 - (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)
```

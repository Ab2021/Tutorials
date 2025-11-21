# Day 26: Semantic Segmentation (FCN & U-Net)

## 1. Problem Definition
**Goal:** Classify **every pixel** in an image into a class (e.g., Road, Car, Sky).
*   **Input:** Image $H \times W \times 3$.
*   **Output:** Mask $H \times W \times C$ (or $H \times W$ with class indices).
*   **Metric:** Mean Intersection over Union (mIoU).

## 2. Fully Convolutional Networks (FCN, 2015)
**Idea:** Adapt classification networks (VGG/ResNet) for dense prediction.
1.  **Convolutionalization:** Replace FC layers with $1 \times 1$ convolutions. Allows input of any size.
2.  **Upsampling:** Use Transposed Convolution (Deconvolution) to recover spatial resolution.
3.  **Skip Connections:** Fuse high-res features (early layers) with low-res semantic features (deep layers) to refine boundaries.
    *   **FCN-32s:** Upsample 32x directly (coarse).
    *   **FCN-8s:** Fuse predictions at 8x scale (fine).

## 3. U-Net (2015)
**Architecture:** Symmetric Encoder-Decoder structure.
*   **Encoder (Contracting Path):** Standard CNN (Conv-Pool). Captures context.
*   **Decoder (Expanding Path):** Upsampling + Convolutions. Precise localization.
*   **Skip Connections:** Concatenate feature maps from Encoder to Decoder at the same level.
    *   Crucial for recovering fine details lost during pooling.

```python
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 1024)
        self.up1 = Up(1024, 512)
        self.up2 = Up(512, 256)
        self.up3 = Up(256, 128)
        self.up4 = Up(128, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x): return self.conv(x)
```

## 4. Loss Functions
*   **Cross-Entropy:** Standard pixel-wise classification.
*   **Dice Loss:** Optimized for overlap. Handles class imbalance well.
    $$ L_{Dice} = 1 - \frac{2 |A \cap B|}{|A| + |B|} $$

## Summary
FCN introduced the idea of end-to-end dense prediction. U-Net refined it with symmetric skip connections, becoming the gold standard for medical imaging and general segmentation.

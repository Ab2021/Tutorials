# Day 123: Semantic Segmentation (UNet)
## Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

---

> **📝 Day 123 Focus:**
> Bounding boxes are boxes. The world is not made of boxes. Roads curve, sidewalks have irregular shapes. **Semantic Segmentation** classifies *every single pixel* in the image. This is crucial for Lane Keeping and Free Space detection.

---

## 🎯 Learning Objectives

By the end of this day, you will be able to:

1.  **Define** Semantic Segmentation vs. Instance Segmentation.
2.  **Explain** the Encoder-Decoder architecture (UNet).
3.  **Implement** Skip Connections to preserve spatial details.
4.  **Construct** a UNet in PyTorch.
5.  **Evaluate** performance using Mean IoU (mIoU).

---

## 📚 Prerequisites & Preparation

### Required Knowledge
-   **Day 121:** CNNs (Conv, Pool).
-   **Transposed Convolution:** Upsampling.

### Hardware Requirements
-   **GPU:** Mandatory for training Segmentation models.

### Software Stack
-   **Python:** `torch`, `torchvision`.

---

## 📖 Theoretical Deep Dive

### 🔹 Part 1: The Task

Input: Image ($H \times W \times 3$).
Output: Mask ($H \times W \times C$), where $C$ is number of classes.
-   Class 0: Road.
-   Class 1: Car.
-   Class 2: Background.

### 🔹 Part 2: The Architecture (Encoder-Decoder)

1.  **Encoder (Downsampling):** Standard CNN (Conv -> Pool). Extracts high-level features ("There is a car"), but loses spatial resolution ("Where exactly?").
2.  **Decoder (Upsampling):** Expands the feature map back to $H \times W$.
3.  **Skip Connections (The "U" in UNet):** Copy high-resolution features from Encoder to Decoder. This recovers the fine details (edges of the road).

### 🔹 Part 3: Upsampling Methods

-   **Bilinear Interpolation:** Simple math. No learnable parameters.
-   **Transposed Convolution (Deconvolution):** Learnable upsampling. Can learn to fill in gaps intelligently.

---

## 💻 Implementation: UNet for Road Segmentation

**Scenario:**
-   Input: $128 \times 128$ Image.
-   Output: Binary Mask (Road vs Not Road).

### 🛠️ Setup
Create `week18_day123` and `unet_seg.py`.

```bash
mkdir -p ~/ros2_ws/src/week18_day123
cd ~/ros2_ws/src/week18_day123
touch unet_seg.py
```

### 👨‍💻 Code: The UNet Model

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """(Conv -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # Encoder
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        # Decoder
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(256, 128) # 256 because 128(up) + 128(skip)
        
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(128, 64) # 128 because 64(up) + 64(skip)
        
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)       # 64 x H x W
        x2 = self.down1(x1)    # 128 x H/2 x W/2
        x3 = self.down2(x2)    # 256 x H/4 x W/4
        
        # Decoder
        x = self.up1(x3)       # 128 x H/2 x W/2
        # Skip Connection: Concatenate x2 and x
        # (Assuming padding kept sizes same. If not, crop x2)
        x = torch.cat([x2, x], dim=1)
        x = self.conv1(x)
        
        x = self.up2(x)        # 64 x H x W
        x = torch.cat([x1, x], dim=1)
        x = self.conv2(x)
        
        logits = self.outc(x)
        return logits

def main():
    # Test Dimensions
    model = UNet(n_channels=3, n_classes=1)
    x = torch.randn(1, 3, 128, 128)
    y = model(x)
    
    print(f"Input Shape: {x.shape}")
    print(f"Output Shape: {y.shape}")
    
    # Check if output size matches input
    assert x.shape[2:] == y.shape[2:]
    print("Test Passed: Dimensions match.")

if __name__ == "__main__":
    main()
```

---

## 🔬 Lab Exercise: The Skip Connection

### Lab Objectives
1.  Run the script.
2.  **Observation:** Output shape matches input shape ($1 \times 1 \times 128 \times 128$).
3.  **Experiment:**
    -   Remove the `torch.cat` (Skip Connection).
    -   Change `DoubleConv` input channels accordingly.
    -   Train on a dummy dataset (e.g., a white circle on black background).
    -   **Result:** Without skip connections, the edges of the circle will be blurry. The network knows "there is a circle roughly here", but lost the exact boundary information during MaxPool.

---

## 🐞 Debugging & Troubleshooting

### Common Issues

#### 1. Size Mismatch in Concatenation
**Symptom:** `RuntimeError: Sizes of tensors must match except in dim 1`.
**Cause:** Convolution without padding reduces size ($128 \to 126$). Pooling halves it ($126 \to 63$). Upsampling doubles it ($63 \to 126$). Original was 128. Mismatch!
**Solution:** Use `padding=1` in all Convs to keep size constant. Or use `F.pad` to match sizes before concat.

#### 2. Class Imbalance
**Symptom:** Model predicts "Background" for everything.
**Cause:** 90% of pixels are background.
**Solution:** Weighted Cross Entropy Loss. Give higher weight to the "Road" class.

---

## ⚡ Optimization & Best Practices

### 1. Dice Loss
Cross Entropy is pixel-wise. It doesn't care about overlap.
**Dice Coefficient:** $2 \times |A \cap B| / (|A| + |B|)$.
-   Use `1 - Dice` as loss function.
-   Better for segmentation than Cross Entropy.

### 2. Cityscapes Dataset
The standard for ADAS segmentation.
-   5000 images of German cities.
-   30 classes (Road, Sidewalk, Person, Car, Sky...).
-   State-of-the-art models (DeepLabV3+) achieve > 80% mIoU.

---

## 🧠 Assessment & Review

### Knowledge Check

1.  **Q:** What is the difference between Semantic and Instance Segmentation?
    *   **A:** Semantic: All cars are "Car" (Red pixels). Instance: Car #1 is Red, Car #2 is Blue.
2.  **Q:** Why do we use $1 \times 1$ convolution at the end?
    *   **A:** To map the feature channels (e.g., 64) to the number of classes (e.g., 1 or 10).
3.  **Q:** What is mIoU?
    *   **A:** Mean Intersection over Union. The average IoU across all classes.

### Challenge Task
**Task:** Binary Mask.
1.  Use the model output `y`.
2.  Apply Sigmoid: `prob = torch.sigmoid(y)`.
3.  Threshold: `mask = prob > 0.5`.
4.  Visualize the mask overlay on the original image.

---

## 📚 Further Reading & References
-   [UNet Paper](https://arxiv.org/abs/1505.04597)
-   [Segmentation Models Pytorch (Library)](https://github.com/qubvel/segmentation_models.pytorch)

---

**Day 123 Complete** | Phase 4: ADAS & Robotics Systems | Week 18: Deep Learning for Perception

# Day 23.2: Semantic Segmentation Architectures (FCN, U-Net, DeepLab) - A Practical Guide

## Introduction: Architectures that Paint by Pixels

To perform semantic segmentation, we need a special kind of CNN architecture that can take an image as input and produce another image (the segmentation mask) of the same height and width as output. Standard classification CNNs are not suitable because their final output is just a single vector of class probabilities.

The dominant architectural pattern for this task is the **encoder-decoder** model. This design, which we first saw in the context of GNNs and Seq2Seq models, is perfectly suited for segmentation. It first down-samples the image to build a rich, semantic understanding (the encoder) and then up-samples it to reconstruct a full-resolution segmentation map (the decoder).

This guide provides a practical overview of the landmark encoder-decoder architectures for semantic segmentation: **FCN**, **U-Net**, and **DeepLab**.

**Today's Learning Objectives:**

1.  **Understand the Fully Convolutional Network (FCN):** Learn how FCNs adapted classification networks for segmentation by replacing fully connected layers with 1x1 convolutions.
2.  **Explore the U-Net Architecture:** Revisit the U-Net and its crucial skip connections that combine deep semantic features with shallow spatial features.
3.  **Learn about Atrous (Dilated) Convolutions:** Understand this key innovation from the DeepLab family that allows for a larger field-of-view without increasing the number of parameters.
4.  **Implement a U-Net from Scratch:** Build a complete U-Net model in PyTorch to see how the encoder, decoder, and skip connections are implemented.

---

## Part 1: The Fully Convolutional Network (FCN)

**The Breakthrough (2015):** FCN was the first work to show that a deep CNN could be trained end-to-end for semantic segmentation.

**The Key Idea:** Take a pre-trained classification network (like VGG) and adapt it for segmentation.
1.  **Make it Convolutional:** The final fully connected layers of the classifier are computationally expensive and discard spatial information. FCN replaces these with standard **1x1 convolutions**. This allows the network to take an input image of any size and produce a low-resolution feature map.
2.  **Up-sample with Transposed Convolutions:** This low-resolution, high-channel feature map is then up-sampled back to the original image size using a single, large **transposed convolution** layer.
3.  **Add Skip Connections:** The authors found that the output from this single up-sampling step was very coarse. To add finer detail, they added skip connections, summing the output of the final layer with the feature maps from earlier, higher-resolution layers in the network (e.g., `pool4` and `pool3`).

**The Legacy:** FCN established the encoder-decoder pattern and the use of skip connections as the foundational approach for modern segmentation.

---

## Part 2: U-Net - The Standard for Biomedical Segmentation

**The Idea (2015):** Developed concurrently with FCN, U-Net took the encoder-decoder idea and made it more systematic and powerful, especially for biomedical images where training data can be scarce.

**The Architecture:**
1.  **Symmetric Encoder-Decoder:** It consists of a **contracting path** (the encoder) and an **expansive path** (the decoder), which gives it its characteristic U-shape.
2.  **Systematic Skip Connections:** The key innovation is the heavy use of skip connections. At each step of the decoder, the up-sampled feature map is **concatenated** with the corresponding feature map from the encoder. 

**Why it Works:** This concatenation provides the decoder with rich, high-resolution spatial information from the encoder at every stage of the up-sampling process. This allows it to reconstruct a very precise, high-resolution segmentation mask, which is why it excels at delineating fine boundaries in medical images.

### 2.1. Implementing a U-Net from Scratch

Let's build a U-Net to see how the pieces fit together.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

print("--- Part 2: Implementing a U-Net ---")

# --- The basic building block: a double convolution ---
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )
    def forward(self, x): return self.double_conv(x)

# --- The U-Net Model ---
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        # --- Encoder (Contracting Path) ---
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(256, 512))

        # --- Decoder (Expansive Path) ---
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv1 = DoubleConv(512, 256) # 256 from up-sample + 256 from skip
        
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv2 = DoubleConv(256, 128) # 128 from up-sample + 128 from skip
        
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv3 = DoubleConv(128, 64) # 64 from up-sample + 64 from skip
        
        # --- Output Layer ---
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        # Decoder with Skip Connections
        u1 = self.up1(x4)
        # Concatenate the skip connection from the encoder
        skip1 = torch.cat([u1, x3], dim=1)
        c1 = self.conv1(skip1)
        
        u2 = self.up2(c1)
        skip2 = torch.cat([u2, x2], dim=1)
        c2 = self.conv2(skip2)
        
        u3 = self.up3(c2)
        skip3 = torch.cat([u3, x1], dim=1)
        c3 = self.conv3(skip3)
        
        # Final output
        logits = self.outc(c3)
        return logits

# --- Dummy Usage ---
model = UNet(n_channels=3, n_classes=10)
input_image = torch.randn(4, 3, 256, 256) # (N, C, H, W)
output_mask_logits = model(input_image)

print(f"U-Net Model Instantiated.")
print(f"Input shape: {input_image.shape}")
print(f"Output logits shape: {output_mask_logits.shape}") # (N, n_classes, H, W)
```

---

## Part 3: DeepLab - Atrous (Dilated) Convolutions

**The Problem:** In a standard encoder, the pooling or strided convolutions rapidly decrease the spatial resolution. This means a lot of detailed spatial information is lost, which the decoder then has to struggle to reconstruct.

**The Solution (DeepLab family): Atrous or Dilated Convolutions**

An atrous convolution is a convolution with "holes." It introduces a new parameter called the **dilation rate**. A dilation rate `r` means the kernel's values are spaced `r-1` pixels apart.

*   **How it works:** It allows the kernel to have a much **larger receptive field** (i.e., see a larger area of the input) **without increasing the number of parameters or decreasing the spatial resolution of the feature map**.
*   **The DeepLab Architecture:** The DeepLab models use a powerful CNN backbone (like a ResNet) but modify it by replacing the last few strided convolutions with atrous convolutions. This allows them to compute rich, semantic features while keeping the feature map relatively large.
*   **Atrous Spatial Pyramid Pooling (ASPP):** To capture context at multiple scales, DeepLab uses an ASPP module. This applies several parallel atrous convolutions with different dilation rates to the same feature map and then concatenates their results.

![Atrous Convolution](https://i.imgur.com/C8w0k2F.gif)

## Conclusion

The encoder-decoder pattern is the dominant paradigm for semantic segmentation. The evolution from FCN to U-Net to DeepLab shows a clear trend towards more sophisticated ways of preserving and combining spatial and semantic information.

**Key Architectural Takeaways:**

1.  **FCN as the Foundation:** It introduced the core idea of adapting classifiers for segmentation by replacing FC layers and using transposed convolutions for up-sampling.
2.  **U-Net for Precision:** Its symmetric architecture with powerful skip connections that concatenate features is the standard for tasks requiring precise boundary delineation, like medical imaging.
3.  **DeepLab for Context:** Its use of atrous convolutions allows it to maintain a large feature map while still having a wide receptive field, and the ASPP module explicitly captures multi-scale context.
4.  **Pre-trained Backbones are Standard:** All modern segmentation models use a powerful, pre-trained classification network (like ResNet) as their encoder to leverage the features learned from large datasets like ImageNet.

By understanding these key architectures, you can choose the right tool for your segmentation task and even begin to design your own custom models.

## Self-Assessment Questions

1.  **FCN:** What was the key innovation of the Fully Convolutional Network that allowed classification networks to be used for segmentation?
2.  **U-Net Skip Connections:** How are the skip connections in a U-Net different from those in the original FCN? (Hint: `concatenate` vs. `sum`).
3.  **Atrous Convolution:** What is the main benefit of using an atrous (dilated) convolution with a rate `r > 1` compared to a standard convolution?
4.  **ASPP:** What is the purpose of the Atrous Spatial Pyramid Pooling (ASPP) module in DeepLab?
5.  **Architectural Choice:** You need to segment very fine, detailed structures in a medical MRI scan. Which architecture, FCN or U-Net, would likely be a better starting point? Why?


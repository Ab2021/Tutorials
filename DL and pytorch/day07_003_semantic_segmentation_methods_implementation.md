# Day 7.3: Semantic Segmentation Methods - A Practical Guide

## Introduction: Painting by Pixels

We have moved from classification (*what*) to object detection (*what and where*). Now, we take the final step in granularity: **Semantic Segmentation**. The goal of semantic segmentation is to assign a class label to **every single pixel** in an image.

Instead of drawing a coarse bounding box around an object, segmentation creates a fine-grained **pixel-level mask**. The output of a segmentation model is an image where each pixel's value corresponds to its predicted class (e.g., all pixels belonging to a "car" are colored blue, all "road" pixels are gray, all "sky" pixels are light blue).

This detailed understanding is crucial for applications like autonomous driving (identifying drivable areas), medical image analysis (delineating tumors or organs), and satellite imagery (classifying land cover).

This guide will explore the architecture of modern segmentation models and provide a practical example using a pre-trained model from `torchvision`.

**Today's Learning Objectives:**

1.  **Understand the Semantic Segmentation Task:** Grasp the concept of pixel-wise classification and the format of the output mask.
2.  **Explore the Encoder-Decoder Architecture:** Learn about this fundamental design pattern used in most segmentation models, where an encoder creates a rich feature representation and a decoder maps it back to the original image resolution.
3.  **Understand Transposed Convolutions:** Learn about the "up-sampling" layers used in the decoder to increase spatial resolution.
4.  **Implement a Basic U-Net Block:** See the core idea of the U-Net architecture, which uses skip connections between the encoder and decoder.
5.  **Use a Pre-trained Segmentation Model:** Learn how to easily load and use a pre-trained DeepLabV3 model for inference.

---

## Part 1: The Encoder-Decoder Architecture

Most modern segmentation networks follow a common architectural pattern: the **encoder-decoder**.

1.  **The Encoder:**
    *   **Purpose:** To extract rich semantic features from the input image. The encoder's job is to understand *what* is in the image.
    *   **Architecture:** This is typically a pre-trained classification network, like a ResNet or a MobileNet, from which the final classification layer has been removed. This is often called the **backbone**.
    *   **Data Flow:** As the image passes through the encoder, its spatial dimensions (`H`, `W`) are progressively reduced (through pooling or strided convolutions), while its channel depth (`C`) is increased. The output is a low-resolution, high-dimensional feature map that contains rich semantic information.

2.  **The Decoder:**
    *   **Purpose:** To take the rich feature map from the encoder and project it back up to the original image resolution, producing the final segmentation mask. The decoder's job is to understand *where* things are, precisely.
    *   **Architecture:** This part of the network progressively **up-samples** the feature map, increasing its spatial dimensions while reducing its channel depth.
    *   **Key Component:** The main tool for up-sampling is the **transposed convolution**.

![Encoder-Decoder](https://i.imgur.com/t1h2O2F.png)

---

## Part 2: The Transposed Convolution (Deconvolution)

A standard convolution takes a large feature map and produces a smaller one. A **transposed convolution** (`nn.ConvTranspose2d`) does the opposite. It takes a small, dense feature map and produces a larger, sparser one. It learns how to "fill in" the details to increase the spatial resolution.

It's often called a "deconvolution," but this term is technically inaccurate. It doesn't reverse the convolution operation; it's simply a convolution that results in up-sampling.

```python
import torch
import torch.nn as nn

print("--- Part 2: Transposed Convolution ---")

# A small, dense feature map (e.g., from an encoder)
input_tensor = torch.randn(1, 16, 8, 8) # (N, C, H, W)

# A transposed convolution to double the height and width
# stride=2 and kernel_size=2 is a common combination for 2x up-sampling.
transposed_conv_layer = nn.ConvTranspose2d(
    in_channels=16, 
    out_channels=8, # We can also change the number of channels
    kernel_size=2, 
    stride=2
)

output_tensor = transposed_conv_layer(input_tensor)

print(f"Input shape: {input_tensor.shape}")
print(f"Output shape after transposed convolution: {output_tensor.shape}") # Expected: (1, 8, 16, 16)
```

---

## Part 3: The U-Net Architecture - Skip Connections for Precision

**The Problem:** The encoder produces a feature map that is semantically rich but spatially poor (it has lost precise location information due to down-sampling). The decoder needs to recover this spatial information.

**The Solution (U-Net):** The **U-Net** architecture, originally developed for biomedical image segmentation, introduced a simple and powerful idea: **skip connections** that connect layers in the encoder directly to corresponding layers in the decoder.

*   **How it works:** The feature map from an early encoder layer (which is spatially rich but semantically poor) is concatenated with the up-sampled feature map in the corresponding decoder layer. This gives the decoder direct access to the high-resolution spatial information it needs to create precise segmentation boundaries.
*   **The "U" Shape:** When drawn, the architecture looks like a letter "U," with the encoder forming the left side, the decoder forming the right side, and the skip connections spanning across the U.

### 3.1. Implementing a U-Net Double Convolutional Block

The core component of a U-Net is a reusable block of two consecutive convolutions.

```python
print("\n--- Part 3: U-Net Architecture ---")

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
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

    def forward(self, x):
        return self.double_conv(x)

# In a full U-Net, you would have an encoder path:
# down1 = DoubleConv(3, 64)
# down2 = DoubleConv(64, 128)
# ...
# And a decoder path with skip connections:
# up1 = Up(1024, 512) # `Up` would contain a ConvTranspose2d
# x = up1(x_from_bottleneck, x_from_encoder_skip_connection)

print("U-Net combines an encoder, a decoder, and skip connections between them.")
print("This allows the decoder to use high-resolution features from the encoder.")
```

---

## Part 4: Using a Pre-trained Segmentation Model

As with other vision tasks, using a pre-trained model is the most effective approach. `torchvision` provides state-of-the-art segmentation models like **DeepLabV3** and **FCN (Fully Convolutional Network)**.

Let's use a pre-trained DeepLabV3 model with a ResNet-101 backbone to segment an image.

```python
import torchvision
from torchvision import transforms
from PIL import Image
import requests
import numpy as np
import matplotlib.pyplot as plt

print("\n--- Part 4: Using a Pre-trained DeepLabV3 ---")

# --- 1. Load the Pre-trained Model ---
model = torchvision.models.segmentation.deeplabv3_resnet101(weights='DEFAULT')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval() # Set to evaluation mode

# --- 2. Load and Transform a Sample Image ---
url = 'https://www.learnpytorch.io/images/computer-vision-pytorch-deeplabv3.png'
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

preprocess = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0).to(device) # Add batch dimension and move to device

# --- 3. Perform Inference ---
with torch.no_grad():
    output = model(input_batch)['out'][0]

# The output is a tensor of shape (num_classes, H, W).
# We take the argmax along the channel dimension to get the predicted class for each pixel.
output_predictions = output.argmax(0)

print(f"Input image size: {image.size}")
print(f"Model output shape (C, H, W): {output.shape}")
print(f"Final prediction mask shape (H, W): {output_predictions.shape}")

# --- 4. Visualize the Results ---
# Create a color palette for the segmentation mask
# PASCAL VOC dataset has 21 classes
palette = torch.tensor([2 ** 25 - 1, 2 ** 15 - 1, 2 ** 21 - 1])
colors = torch.as_tensor([i for i in range(21)])[:, None] * palette
colors = (colors % 255).numpy().astype("uint8")

# Convert the prediction mask to an RGB image
r = Image.fromarray(output_predictions.byte().cpu().numpy()).resize(image.size)
r.putpalette(colors)

# Overlay the mask on the original image
blended_image = Image.blend(image, r.convert('RGB'), alpha=0.6)

# --- Plotting ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(image)
axs[0].set_title('Original Image')
axs[1].imshow(r)
axs[1].set_title('Segmentation Mask')
axs[2].imshow(blended_image)
axs[2].set_title('Blended Image')
for ax in axs: ax.axis('off')
plt.show()
```

## Conclusion

Semantic segmentation provides the most detailed level of scene understanding by assigning a class to every pixel. The dominant architectural pattern for this task is the **encoder-decoder** network.

**Key Architectural Takeaways:**

1.  **Encoder-Decoder Structure:** An encoder (backbone) creates a low-resolution, high-dimensional feature map, and a decoder up-samples this map to produce a full-resolution segmentation mask.
2.  **Up-sampling with Transposed Convolutions:** The `nn.ConvTranspose2d` layer is the primary tool used in the decoder to increase spatial resolution.
3.  **U-Net and Skip Connections:** The U-Net architecture introduced the critical idea of using skip connections to feed high-resolution spatial information from the encoder directly to the decoder, leading to much more precise segmentation boundaries.
4.  **Pre-trained Models are Key:** As with other vision tasks, using a powerful, pre-trained segmentation model like DeepLabV3 is the most practical and effective approach.

With this understanding, you can now tackle problems that require a fine-grained, pixel-level understanding of the world.

## Self-Assessment Questions

1.  **Segmentation vs. Detection:** What is the main difference between the output of an object detection model and a semantic segmentation model?
2.  **Encoder vs. Decoder:** What is the primary role of the encoder in a segmentation network? What about the decoder?
3.  **Transposed Convolution:** If you have a tensor of shape `[1, 64, 16, 16]` and you pass it through an `nn.ConvTranspose2d` layer with `stride=2`, what will the height and width of the output be?
4.  **Skip Connections:** Why are the skip connections in a U-Net so important for producing accurate segmentation masks?
5.  **Output Shape:** The output of the DeepLabV3 model has a shape of `(N, C, H, W)`, where `C` is the number of classes. How do you convert this to the final 2D segmentation mask of shape `(H, W)`?

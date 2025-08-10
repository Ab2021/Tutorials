# Day 18.1: CNN Fundamentals & Convolution Theory - A Practical Deep Dive

## Introduction: Learning Spatial Hierarchies

Convolutional Neural Networks (CNNs) are the cornerstone of modern computer vision. Their design is inspired by the human visual cortex and is built on a powerful premise: learning a hierarchy of spatial features. Instead of learning from flattened vectors, CNNs process data in its grid-like topology, allowing them to effectively learn patterns from images, videos, and other spatial data.

The fundamental operation that enables this is the **convolution**. By sliding small filters (kernels) over the input image, a CNN learns to detect simple features like edges and corners in its initial layers. Subsequent layers then combine these simple features to detect more complex patterns like textures, shapes, and eventually, whole objects.

This guide provides a deep, practical dive into the convolution operation itself, breaking down its mechanics and visualizing its effects to build a strong, foundational intuition.

**Today's Learning Objectives:**

1.  **Revisit the Convolution Operation:** Understand and visualize how a kernel (filter) slides over an input image to produce a feature map.
2.  **Master the Key Parameters:** Gain a deep, practical understanding of **Channels (Depth)**, **Kernel Size**, **Stride**, and **Padding**.
3.  **Calculate Output Dimensions:** Learn the formula to precisely calculate the spatial dimensions of the output feature map.
4.  **Implement Convolutions in PyTorch:** Use `torch.nn.functional.conv2d` to manually apply filters and see their effects.
5.  **Connect Theory to `nn.Conv2d`:** Understand how the `nn.Conv2d` layer encapsulates the convolution operation into a learnable module.

---

## Part 1: The Convolution Operation in Detail

A 2D convolution is a simple element-wise multiplication and sum.

1.  **The Kernel:** We have a small matrix called a kernel or filter (e.g., 3x3).
2.  **Placement:** We place this kernel over a patch of the input image.
3.  **Element-wise Multiplication:** We multiply the values in the kernel by the corresponding pixel values in the image patch.
4.  **Summation:** We sum up all the results of the multiplication.
5.  **Output Pixel:** This single sum becomes the value of one pixel in the output **feature map**.
6.  **Sliding:** We slide the kernel over to the next position and repeat the process until the entire image has been covered.

![Convolution GIF](https://i.imgur.com/VGoaZ3A.gif)

### 1.1. The Role of Channels (Depth)

Modern images are not 2D; they have a depth dimension (e.g., 3 channels for RGB). The convolution operation naturally extends to this.

*   An input image has shape `(C_in, H, W)`.
*   The convolutional kernel must have the **same depth** as the input. So, a kernel applied to an RGB image will have a shape of `(C_in, K_h, K_w)`, e.g., `(3, 3, 3)`.
*   The convolution is performed by sliding this 3D kernel over the 3D input. The element-wise multiplication and sum happens across all three dimensions (`C_in`, `K_h`, `K_w`).
*   The result is still a **single number** for each position, producing a 2D feature map.
*   If we want to produce multiple output channels (`C_out`), we simply use `C_out` different filters. Each filter learns to detect a different feature, producing its own unique feature map. These maps are then stacked to form the final output tensor of shape `(C_out, H_out, W_out)`.

---

## Part 2: The Parameters of Convolution

### 2.1. Kernel Size

*   **What it is:** The height and width of the filter. Common sizes are 3x3 and 5x5.
*   **Effect:** Determines the **receptive field** of the neurons in the output feature map. A 3x3 kernel means each output pixel is a function of a 3x3 patch of input pixels.

### 2.2. Padding

*   **The Problem:** A 3x3 kernel can't be centered on the pixels at the very edge of an image. This means the output feature map will be smaller than the input image, and the edge pixels are processed less often than the center pixels.
*   **The Solution:** **Padding** adds a border of zeros around the input image. This allows the kernel to be centered on every pixel, including the original edge pixels.
*   **Common Practice:** For a kernel of size `K`, using `padding = (K-1)/2` (e.g., `padding=1` for a 3x3 kernel) will result in an output feature map with the **same height and width** as the input (assuming a stride of 1). This is often called "same" padding.

### 2.3. Stride

*   **What it is:** The step size the kernel takes as it slides across the image.
*   **Effect:** A stride greater than 1 will cause the kernel to skip pixels, resulting in **down-sampling**. A `stride=2` will produce an output feature map that is roughly half the size of the input.
*   **Usage:** While `MaxPool` is a common way to down-sample, using a strided convolution is another powerful and common technique.

### 2.4. The Output Size Formula

You can precisely calculate the output height or width with this formula:

`Output_size = floor( (Input_size - Kernel_size + 2 * Padding) / Stride ) + 1`

---

## Part 3: Visualizing Convolutions in PyTorch

Let's use `F.conv2d` to see the effects of these parameters in code.

```python
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

print("--- Part 3: Visualizing Convolutions ---")

# --- Create a simple dummy image (a white cross on a black background) ---
image = torch.zeros(1, 1, 10, 10) # (N, C_in, H, W)
image[:, :, 4:6, :] = 1 # Horizontal bar
image[:, :, :, 4:6] = 1 # Vertical bar

# --- Define some kernels ---
# 1. Vertical Edge Detector
kernel_v = torch.tensor([[[[ 1, 0, -1],
                           [ 1, 0, -1],
                           [ 1, 0, -1]]]], dtype=torch.float32)

# 2. Horizontal Edge Detector
kernel_h = torch.tensor([[[[ 1,  1,  1],
                           [ 0,  0,  0],
                           [-1, -1, -1]]]], dtype=torch.float32)

# --- Apply convolutions and visualize ---
def apply_and_show(image, kernel, title):
    # Apply the convolution
    output = F.conv2d(image, kernel, padding=1)
    
    # Visualize
    fig, axs = plt.subplots(1, 2, figsize=(8, 4))
    axs[0].imshow(image.squeeze(), cmap='gray')
    axs[0].set_title('Original Image')
    axs[1].imshow(output.squeeze(), cmap='gray')
    axs[1].set_title(title)
    for ax in axs: ax.axis('off')
    plt.show()

print("Applying a vertical edge detection kernel...")
apply_and_show(image, kernel_v, "Vertical Edge Feature Map")

print("\nApplying a horizontal edge detection kernel...")
apply_and_show(image, kernel_h, "Horizontal Edge Feature Map")

# --- Demonstrate Stride ---
print("\nDemonstrating the effect of Stride...")
output_stride2 = F.conv2d(image, kernel_v, stride=2, padding=1)
print(f"Original image shape: {image.shape}")
print(f"Output shape with stride=2: {output_stride2.shape}")
```

**Interpretation:**
*   The vertical edge detector produces high positive values on the left edges of the cross and high negative values on the right edges, and zero elsewhere.
*   The horizontal edge detector does the same for the top and bottom edges.
*   This is exactly what a CNN learns to do in its first layer: create a bank of filters that detect basic features like edges in different orientations.

---

## Part 4: The `nn.Conv2d` Layer

The `nn.Conv2d` module encapsulates this operation into a learnable layer.

*   When you create `nn.Conv2d(in_channels, out_channels, ...)`:
    *   PyTorch automatically creates a `weight` tensor of shape `(out_channels, in_channels, kernel_height, kernel_width)`. This is the bank of learnable filters.
    *   It also creates a `bias` tensor of shape `(out_channels)`. Each output channel gets its own bias term.
*   During training, `loss.backward()` computes the gradients with respect to both the `weight` and `bias` tensors, and the optimizer updates them.

```python
import torch.nn as nn

print("\n--- Part 4: The nn.Conv2d Layer ---")

# --- Parameters ---
in_channels = 3  # Input is an RGB image
out_channels = 16 # We want to learn 16 different features
kernel_size = 5

# --- Create the layer ---
conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=2)

print(f"Created an nn.Conv2d layer.")
print(f"  - Learnable weight matrix (filters) shape: {conv_layer.weight.shape}")
print(f"  - Learnable bias vector shape: {conv_layer.bias.shape}")

# --- Apply the layer ---
input_image_batch = torch.randn(8, 3, 64, 64) # (N, C, H, W)
output_feature_maps = conv_layer(input_image_batch)

print(f"\nInput batch shape: {input_image_batch.shape}")
print(f"Output feature maps shape: {output_feature_maps.shape}")
```

## Conclusion

The convolution is a simple yet profoundly powerful operation. It allows a network to learn and apply spatial filters across an entire image using a very small number of shared parameters. This ability to learn a hierarchy of features, from simple edges in the first layer to complex object parts in deeper layers, is the fundamental reason for the success of CNNs.

**Key Takeaways:**

1.  **Convolution is Filtering:** The operation is equivalent to sliding a filter (kernel) over an image to produce a feature map that highlights certain patterns.
2.  **Channels, Kernel, Padding, Stride:** These four parameters give you complete control over the convolution operation and the dimensions of the output.
3.  **The Output Size Formula is Your Friend:** Use it to design your architectures and ensure the tensor dimensions line up correctly between layers.
4.  **CNNs Learn the Filters:** The magic of a `nn.Conv2d` layer is that the values in the kernels are not hand-crafted; they are learned automatically through backpropagation to be the most useful filters for the given task.

With this deep, practical understanding of the convolution operation, you are now ready to build and analyze complex, state-of-the-art CNN architectures.

## Self-Assessment Questions

1.  **Channels:** If your input image has 3 channels (RGB) and your first `nn.Conv2d` layer has `out_channels=32`, what is the shape of the layer's `weight` tensor (assuming a 3x3 kernel)?
2.  **Padding:** You have a 32x32 input image and you apply a 3x3 convolution with `stride=1`. What `padding` value should you use to ensure the output is also 32x32?
3.  **Stride:** You have a 64x64 input image. You apply a convolution with `stride=2`. What will be the approximate size of the output feature map?
4.  **Receptive Field:** Does a larger kernel size lead to a larger or smaller receptive field for the neurons in the output map?
5.  **Learned vs. Manual:** What is the key difference between the Sobel filter we created manually and a filter learned by a `nn.Conv2d` layer?

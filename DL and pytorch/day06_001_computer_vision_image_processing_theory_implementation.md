# Day 6.1: Computer Vision & Image Processing Theory - A Practical Introduction

## Introduction: Seeing the World in Pixels

Before we can build complex Convolutional Neural Networks (CNNs) to understand images, it's essential to grasp the fundamentals of how images are represented and manipulated digitally. Computer Vision (CV) is not just about deep learning; it's built on a rich history of image processing techniques that are still relevant today and provide the intuition for why modern architectures work the way they do.

This guide will be a practical exploration of core image processing concepts. We will use PyTorch and `torchvision` not for training models, but as a powerful scientific computing library to load, manipulate, and visualize images. By seeing how these classical techniques work, you will gain a much deeper understanding of what a CNN is actually learning to do automatically.

**Today's Learning Objectives:**

1.  **Understand Digital Image Representation:** See how images are just tensors of numbers and how color spaces (RGB, Grayscale) work.
2.  **Analyze Image Properties with Histograms:** Learn to compute and interpret histograms to understand the distribution of pixel intensities.
3.  **Master Image Filtering with Convolutions:** Manually apply convolution operations with different kernels (filters) to achieve effects like blurring and sharpening.
4.  **Detect Edges:** Understand and implement edge detection, a fundamental step in identifying objects, using filters like the Sobel operator.
5.  **Connect Classical CV to Modern Deep Learning:** See how the concept of a manually designed "filter" in classical CV is analogous to a learned "filter" in a CNN's convolutional layer.

---

## Part 1: The Digital Image as a Tensor

An image is a grid of pixels. Each pixel has a value representing its intensity or color.

*   **Grayscale Image:** A 2D tensor (or matrix) of shape `(Height, Width)`. Each pixel is a single scalar value, typically from 0 (black) to 255 (white).
*   **Color Image (RGB):** A 3D tensor of shape `(Height, Width, Channels)` or, more commonly in PyTorch, `(Channels, Height, Width)`. The three channels correspond to the intensity of Red, Green, and Blue light.

```python
import torch
from torchvision import transforms
from PIL import Image
import requests
import matplotlib.pyplot as plt
import numpy as np

print("--- Part 1: The Digital Image as a Tensor ---")

# --- Load a sample image ---
url = 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
image_pil = Image.open(requests.get(url, stream=True).raw).convert("RGB")

# --- Convert to a PyTorch Tensor ---
# ToTensor() converts a PIL Image (H, W, C) [0, 255] to a Tensor (C, H, W) [0.0, 1.0]
to_tensor = transforms.ToTensor()
image_tensor = to_tensor(image_pil)

print(f"Original PIL Image size: {image_pil.size}")
print(f"Tensor shape (C, H, W): {image_tensor.shape}")

# --- Separate the color channels ---
red_channel = image_tensor[0]
green_channel = image_tensor[1]
blue_channel = image_tensor[2]

# --- Convert to Grayscale ---
# torchvision has a transform for this.
# It uses the standard formula: L = 0.299*R + 0.587*G + 0.114*B
to_grayscale = transforms.Grayscale()
gayscale_pil = to_grayscale(image_pil)
gayscale_tensor = to_tensor(grayscale_pil)

# --- Visualize ---
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
axs[0].imshow(image_pil)
axs[0].set_title("Original RGB Image")
axs[0].axis('off')

axs[1].imshow(grayscale_pil, cmap='gray')
axs[1].set_title("Grayscale Image")
axs[1].axis('off')

# Display the red channel
axs[2].imshow(red_channel.numpy(), cmap='Reds')
axs[2].set_title("Red Channel")
axs[2].axis('off')

plt.show()
```

---

## Part 2: Analyzing Images with Histograms

A histogram shows the distribution of pixel intensity values in an image. It's a bar graph where the x-axis is the pixel value (e.g., 0-255) and the y-axis is the number of pixels with that value.

Histograms are useful for understanding an image's brightness, contrast, and overall tonal range.

```python
print("\n--- Part 2: Image Histograms ---")

# --- Calculate Histogram for the Grayscale Image ---
# We flatten the 2D image tensor into a 1D vector to compute the histogram.
# We multiply by 255 to get back to the [0, 255] range for easier interpretation.
histogram_data = grayscale_tensor.flatten() * 255

plt.figure(figsize=(10, 6))
plt.hist(histogram_data.numpy(), bins=256, range=(0, 255), color='gray', alpha=0.8)
plt.title('Pixel Intensity Histogram for Grayscale Image')
plt.xlabel('Pixel Intensity (0-255)')
plt.ylabel('Frequency (Number of Pixels)')
plt.grid(True)
plt.show()

# --- Histogram Equalization (A common technique) ---
# This technique spreads out the most frequent intensity values to enhance contrast.
from torchvision.transforms.functional import equalize

equalized_tensor = equalize( (grayscale_tensor * 255).to(torch.uint8) )
equalized_hist_data = equalized_tensor.flatten()

fig, axs = plt.subplots(1, 2, figsize=(15, 5))
axs[0].imshow(equalized_tensor.squeeze(), cmap='gray')
axs[0].set_title('Equalized Image')
axs[0].axis('off')
axs[1].hist(equalized_hist_data.numpy(), bins=256, range=(0, 255), color='gray', alpha=0.8)
axs[1].set_title('Histogram of Equalized Image')
plt.show()
```

---

## Part 3: The Convolution Operation - Image Filtering

A convolution is the fundamental operation of image filtering and CNNs. It involves sliding a small matrix, called a **kernel** or **filter**, over the image. At each position, we compute the element-wise product of the kernel and the overlapping image patch, and then sum up the results to get the new pixel value.

This simple operation can produce a wide range of effects, like blurring, sharpening, and edge detection, depending on the values in the kernel.

We can implement this using `torch.nn.functional.conv2d`.

```python
import torch.nn.functional as F

print("\n--- Part 3: Image Filtering with Convolutions ---")

# The input to conv2d must be a batch of images, so we add a batch dimension.
# Input shape: (N, C_in, H, W)
input_image = grayscale_tensor.unsqueeze(0) # Add batch dimension

# --- 1. The Identity Kernel ---
# This kernel does nothing, the output is the same as the input.
identity_kernel = torch.tensor([[[[0, 0, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)

# --- 2. The Box Blur Kernel ---
# This kernel averages the neighboring pixels, resulting in a blurring effect.
box_blur_kernel = torch.ones((1, 1, 3, 3), dtype=torch.float32) / 9.0

# --- 3. The Sharpen Kernel ---
# This kernel emphasizes differences between a pixel and its neighbors.
sharpen_kernel = torch.tensor([[[[ 0, -1,  0], [-1,  5, -1], [ 0, -1,  0]]]], dtype=torch.float32)

# --- Apply the convolutions ---
identity_output = F.conv2d(input_image, identity_kernel, padding=1)
blur_output = F.conv2d(input_image, box_blur_kernel, padding=1)
sharpen_output = F.conv2d(input_image, sharpen_kernel, padding=1)

# --- Visualize ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(grayscale_tensor.squeeze(), cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(identity_output.squeeze(), cmap='gray')
axs[1].set_title('Identity Filter')
axs[2].imshow(blur_output.squeeze(), cmap='gray')
axs[2].set_title('Box Blur Filter')
axs[3].imshow(sharpen_output.squeeze(), cmap='gray')
axs[3].set_title('Sharpen Filter')
for ax in axs: ax.axis('off')
plt.show()
```

---

## Part 4: Edge Detection

Edge detection is a crucial first step in many CV pipelines. Edges correspond to sharp changes in pixel intensity and often represent the boundaries of objects.

The **Sobel operator** is a classic edge detection algorithm. It uses two separate kernels: one to detect horizontal edges (`Gx`) and one to detect vertical edges (`Gy`).

```python
print("\n--- Part 4: Edge Detection with Sobel Filter ---")

# --- Define the Sobel Kernels ---
# Gx: Detects vertical lines
sobel_x_kernel = torch.tensor([[[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]]], dtype=torch.float32)

# Gy: Detects horizontal lines
sobel_y_kernel = torch.tensor([[[[-1, -2, -1], [0, 0, 0], [1, 2, 1]]]], dtype=torch.float32)

# --- Apply the convolutions ---
# Note: input_image is still the grayscale tensor with a batch dimension
edges_x = F.conv2d(input_image, sobel_x_kernel, padding=1)
edges_y = F.conv2d(input_image, sobel_y_kernel, padding=1)

# --- Combine the results ---
# The final edge magnitude is the square root of the sum of the squares of Gx and Gy.
edge_magnitude = torch.sqrt(edges_x**2 + edges_y**2)

# --- Visualize ---
fig, axs = plt.subplots(1, 4, figsize=(20, 5))
axs[0].imshow(grayscale_tensor.squeeze(), cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(edges_x.squeeze(), cmap='gray')
axs[1].set_title('Sobel X (Vertical Edges)')
axs[2].imshow(edges_y.squeeze(), cmap='gray')
axs[2].set_title('Sobel Y (Horizontal Edges)')
axs[3].imshow(edge_magnitude.squeeze(), cmap='gray')
axs[3].set_title('Edge Magnitude')
for ax in axs: ax.axis('off')
plt.show()
```

## Conclusion: From Manual Filters to Learned Features

We have seen how we can manually design kernels to perform specific tasks like blurring, sharpening, and edge detection. This is the essence of classical computer vision.

**The Deep Learning Connection:**

A Convolutional Neural Network (CNN) operates on the exact same principle of convolution. However, there is one profound difference: **the values in the kernels are not pre-defined. They are learned.**

*   In a `nn.Conv2d` layer, the `weight` tensor is the collection of kernels.
*   These weights are initialized randomly.
*   During training, through backpropagation, the network learns the optimal values for these kernels to solve the given task.

In the early layers of a CNN trained for image classification, you will often find that the network has automatically learned to create kernels that look just like Gabor filters and Sobel filters—it has learned that detecting edges and textures is a fundamental and useful first step. Deeper layers then combine these simple features to detect more complex patterns like eyes, wheels, or text.

By understanding the classical techniques, you gain a powerful intuition for what a CNN is doing under the hood. It is, in essence, a machine for automatically discovering the most useful set of hierarchical filters for a specific visual task.

## Self-Assessment Questions

1.  **Image Representation:** What is the standard tensor shape for a batch of color images in PyTorch?
2.  **Histograms:** If an image histogram is heavily skewed to the left (i.e., most pixels have low intensity values), what would you expect the image to look like?
3.  **Convolution:** If you apply a 3x3 blurring kernel to a 28x28 image with `padding=0`, what will be the height and width of the output image?
4.  **Sobel Operator:** The Sobel operator uses two kernels. What is the purpose of each one?
5.  **The Learning Connection:** What is the key difference between the sharpen kernel we defined manually and a kernel in an `nn.Conv2d` layer?

```
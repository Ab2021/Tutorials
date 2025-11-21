# Day 15: Segmentation - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Transposed Convolution, Dilated Conv, and DeepLab

## 1. Transposed Convolution (Deconvolution)

How to go from $16 \times 16$ to $32 \times 32$?
**Transposed Convolution** learns a kernel that broadcasts input pixels to a larger area.
*   Input: $2 \times 2$. Kernel: $3 \times 3$. Stride: 2.
*   It inserts zeros between input pixels (dilation) and then convolves.
*   **Problem**: Checkerboard Artifacts. If kernel size is not divisible by stride, overlaps are uneven.

## 2. Bilinear Upsampling + Convolution

The modern alternative to Transposed Conv.
1.  **Upsample**: Resize using standard Bilinear Interpolation (no params).
2.  **Convolution**: Refine the result with a $3 \times 3$ conv.
*   Smooth, artifact-free, and cheaper.

## 3. Dilated (Atrous) Convolution

In DeepLab, instead of Downsampling (losing info) and then Upsampling (guessing info), we keep the resolution high.
*   We remove MaxPool layers.
*   We use **Dilated Convolutions** to increase Receptive Field without downsampling.
*   Result: High-resolution feature maps ($H/8$ instead of $H/32$).

## 4. DeepLab & ASPP

**Atrous Spatial Pyramid Pooling (ASPP)**:
*   Apply multiple dilated convolutions with different rates ($r=6, 12, 18$) in parallel.
*   Captures objects at multiple scales (small, medium, large) simultaneously.
*   Concatenate results.

## 5. Panoptic Segmentation

Combines Semantic (Stuff) and Instance (Things).
*   **Stuff**: Sky, Road, Grass (Amorphous regions).
*   **Things**: Car, Person, Dog (Countable objects).
*   Architecture: Shared Backbone $\to$ Semantic Head + Instance Head (Mask R-CNN).

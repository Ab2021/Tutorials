# Day 11: Convolutions - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Im2Col, Winograd, and Grouped Convolutions

## 1. How Convolutions are actually implemented (GEMM)

Naive sliding window loops are slow (poor cache locality).
Frameworks convert Convolution into Matrix Multiplication (GEMM).

### Im2Col (Image to Column)
1.  **Unroll**: Extract each $K \times K$ patch from the image and flatten it into a column.
    *   Input Image: $(C, H, W)$.
    *   Col Matrix: $(C \times K^2, H_{out} \times W_{out})$.
2.  **Flatten Kernels**: Reshape weights into a matrix.
    *   Weights: $(C_{out}, C \times K^2)$.
3.  **MatMul**: $W \times Col$.
    *   Result: $(C_{out}, H_{out} \times W_{out})$.
4.  **Reshape**: Fold back to $(C_{out}, H_{out}, W_{out})$.

*   **Pros**: Uses highly optimized BLAS (cuBLAS).
*   **Cons**: Increases memory usage (duplicated pixels in overlap).

## 2. Grouped Convolutions

Standard Conv connects every input channel to every output channel.
*   Params: $C_{out} \times C_{in} \times K^2$.
*   FLOPs: $H \times W \times C_{out} \times C_{in} \times K^2$.

**Grouped Conv**: Split channels into $G$ groups.
*   Group 1 inputs connect only to Group 1 outputs.
*   Params: $C_{out} \times (C_{in}/G) \times K^2$.
*   Reduces parameters and FLOPs by factor of $G$.
*   **Extreme Case**: Depthwise Convolution ($G = C_{in}$).

```python
# AlexNet used groups=2 to split model across 2 GPUs
# ResNeXt uses groups=32
conv = nn.Conv2d(64, 64, 3, groups=32)
```

## 3. Depthwise Separable Convolution

The building block of MobileNets.
Decomposes standard convolution into:
1.  **Depthwise**: Spatial filtering ($3 \times 3$, $G=C_{in}$).
2.  **Pointwise**: Channel mixing ($1 \times 1$, standard).

*   Cost reduction: $\approx \frac{1}{K^2}$.
*   Allows running heavy models on mobile devices.

## 4. 1x1 Convolution (Pointwise)

Why use a $1 \times 1$ kernel?
*   **Dimensionality Reduction**: Reduce channels from 256 to 64 (Bottleneck layer).
*   **Non-Linearity**: Adds ReLU without changing spatial dimensions.
*   **Cross-Channel Interaction**: "MLP per pixel".

## 5. Transposed Convolution (Deconvolution)

Used for Upsampling (GANs, Segmentation).
It is **not** the inverse of convolution. It swaps forward and backward passes of a convolution.
*   Often causes "Checkerboard Artifacts".
*   **Better Alternative**: `nn.Upsample(scale_factor=2, mode='bilinear')` + `nn.Conv2d`.

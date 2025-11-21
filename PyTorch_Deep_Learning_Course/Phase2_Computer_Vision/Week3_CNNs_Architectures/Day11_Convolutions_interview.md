# Day 11: Convolutions - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: CNN Mechanics, Efficiency, and Math

### 1. Calculate the output size of a Convolution layer.
**Answer:**
$$ H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} + 1 \rfloor $$
*   Example: Input 224, Kernel 7, Stride 2, Padding 3.
*   $(224 + 6 - 7) / 2 + 1 = 223 / 2 + 1 = 111 + 1 = 112$.

### 2. What is the difference between Correlation and Convolution?
**Answer:**
*   Mathematically, Convolution flips the kernel. Correlation does not.
*   In Deep Learning, we strictly use **Cross-Correlation** (no flip).
*   We call it "Convolution" by convention. Since weights are learned, the flip doesn't matter (the network learns the flipped weights if needed).

### 3. Why do we use $3 \times 3$ kernels instead of $5 \times 5$ or $7 \times 7$?
**Answer:**
*   **Parameter Efficiency**: Two stacked $3 \times 3$ layers have RF of $5 \times 5$.
    *   Params: $2 \times (3^2 C^2) = 18 C^2$.
    *   Single $5 \times 5$: $1 \times (5^2 C^2) = 25 C^2$.
*   **Non-Linearity**: Two layers have 2 ReLUs, making the function more discriminative.

### 4. What is a "Receptive Field"? How do you calculate it?
**Answer:**
*   The region of the input image that affects a specific output pixel.
*   $RF_{l} = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$.
*   It grows exponentially with depth if strides > 1.

### 5. Explain "Dilated Convolution".
**Answer:**
*   Inserting holes (zeros) between kernel elements.
*   Increases Receptive Field without increasing parameters or losing resolution (no downsampling).
*   Crucial for Semantic Segmentation (DeepLab) to capture global context.

### 6. What is "Global Average Pooling"?
**Answer:**
*   Averaging the entire feature map $(H \times W)$ into a single number.
*   Used at the end of CNNs to replace Fully Connected layers.
*   Reduces parameters, prevents overfitting, allows variable input size.

### 7. Why are 1x1 Convolutions useful?
**Answer:**
*   **Channel Reduction**: Reducing 512 channels to 64 (Bottleneck).
*   **Efficiency**: Makes computation cheaper for subsequent $3 \times 3$ layers.
*   Used in Inception and ResNet Bottlenecks.

### 8. What is "Depthwise Separable Convolution"?
**Answer:**
*   Factorizing standard conv into Depthwise (spatial) + Pointwise (channel).
*   Drastically reduces parameters and FLOPs.
*   Used in MobileNet.

### 9. What are "Checkerboard Artifacts" in Transposed Convolutions?
**Answer:**
*   Caused when the kernel size is not divisible by the stride (overlap issues).
*   The upsampled image has a grid-like pattern of intensity variations.
*   Fix: Use Bilinear Upsampling + Convolution.

### 10. How does "Im2Col" work?
**Answer:**
*   Converts the 3D convolution operation into a 2D Matrix Multiplication.
*   Duplicates memory (patches overlap) but allows using highly optimized BLAS libraries (GEMM).

### 11. What is the "Translation Equivariance" of CNNs?
**Answer:**
*   If you shift the input object, the activation map shifts by the same amount.
*   This allows the network to detect the object regardless of its position.
*   Note: CNNs are *not* rotation or scale equivariant by default.

### 12. Why is Padding important?
**Answer:**
*   **Shape Preservation**: Keeps output size same as input (Same Padding).
*   **Border Information**: Without padding, border pixels are used less in convolutions than center pixels. Padding allows kernels to "see" the edge.

### 13. What is "Strided Convolution" vs "Pooling"?
**Answer:**
*   Both downsample the image.
*   **Pooling**: Fixed operation (Max/Avg). No parameters. Discards information.
*   **Strided Conv**: Learnable downsampling. The network learns *how* to summarize the patch.
*   Modern trend: Prefer Strided Conv (All-Convolutional Net).

### 14. How many parameters in a Conv Layer?
**Answer:**
*   Weights: $C_{out} \times C_{in} \times K \times K$.
*   Bias: $C_{out}$.
*   Note: Input H/W does not affect parameter count!

### 15. What is "Grouped Convolution"?
**Answer:**
*   Splitting channels into independent groups.
*   Reduces computation.
*   Originally used in AlexNet to fit on 2 GPUs.
*   Now used in ResNeXt for better accuracy/parameter trade-off.

### 16. What is the "Bottleneck" design in ResNet?
**Answer:**
*   $1 \times 1$ (reduce dim) $\to$ $3 \times 3$ (conv) $\to$ $1 \times 1$ (restore dim).
*   Allows the $3 \times 3$ conv to operate on fewer channels, saving compute.

### 17. Explain "Spatially Separable Convolution".
**Answer:**
*   Decomposing $3 \times 3$ into $3 \times 1$ followed by $1 \times 3$.
*   Reduces params from 9 to 6.
*   Used in Inception V3.
*   Less common on GPU because $3 \times 1$ kernels are memory inefficient.

### 18. What is "Deformable Convolution"?
**Answer:**
*   Learning 2D offsets for each grid position in the kernel.
*   The kernel can change shape to fit the object (e.g., curve around a bird).
*   Improves geometric transformation modeling.

### 19. Why do we increase channels as we go deeper?
**Answer:**
*   **Spatial Resolution decreases**: We lose spatial info ($H, W$).
*   **Semantic Information increases**: We need more channels ($C$) to store complex, high-level features (eyes, wheels, faces).
*   Keeps computational cost roughly balanced across layers.

### 20. What is the difference between `Valid` and `Same` padding?
**Answer:**
*   **Valid**: No padding. Output size reduces.
*   **Same**: Padding added so output size equals input size (assuming stride 1).

# Day 26 Deep Dive: Upsampling & Dice Loss

## 1. Upsampling Techniques
How to increase spatial resolution?
1.  **Nearest Neighbor:** Copy values. Simple, no parameters.
2.  **Bilinear Interpolation:** Weighted average of neighbors. Smooth.
3.  **Transposed Convolution (Deconvolution):**
    *   Learnable upsampling.
    *   Inserts zeros between pixels and convolves with a kernel.
    *   **Checkerboard Artifacts:** Common issue if kernel size is not divisible by stride.
4.  **Pixel Shuffle:** Used in Super-Resolution. Reshapes channels into space ($H \times W \times r^2 C \to rH \times rW \times C$).

## 2. Dice Loss vs Cross-Entropy
**Scenario:** Small tumor (1% of pixels) in large image.
*   **Cross-Entropy:** Model predicts "Background" everywhere. Accuracy = 99%. Loss is low.
*   **Dice Loss:**
    *   Intersection = 0.
    *   Dice Score = 0. Loss = 1 (Max).
    *   Forces model to predict the tumor.
    *   **Soft Dice:** Uses probabilities instead of binary masks for differentiability.
    $$ L = 1 - \frac{2 \sum p_i g_i}{\sum p_i^2 + \sum g_i^2} $$

## 3. The "Valid" Padding Issue
*   Original U-Net used unpadded convolutions, causing output size < input size.
*   Required "Overlap-Tile Strategy" for large images.
*   **Modern U-Net:** Uses `padding=1` to keep size constant ($H \times W \to H \times W$). Much easier to handle.

## 4. SegNet (2015)
**Idea:** Store **Max-Pooling Indices** in Encoder.
*   Use these indices in Decoder for upsampling (Unpooling).
*   **Pros:** No need to store full feature maps (memory efficient).
*   **Cons:** Slightly worse performance than U-Net (concatenation is better than unpooling).

## Summary
Choosing the right upsampling method (Bilinear vs Transposed) and Loss function (Dice vs CE) is critical for segmentation, especially with imbalanced classes.

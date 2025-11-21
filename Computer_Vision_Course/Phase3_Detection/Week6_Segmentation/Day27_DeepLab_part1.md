# Day 27 Deep Dive: CRF & PSPNet

## 1. Conditional Random Fields (CRF)
Used in DeepLab v1/v2 as post-processing.
*   **Idea:** Model pixel labels as a probabilistic graph.
*   **Energy Function:**
    $$ E(x) = \sum \psi_u(x_i) + \sum \psi_p(x_i, x_j) $$
    *   **Unary Potential ($\psi_u$):** Probability from CNN (Pixel $i$ looks like a "Cat").
    *   **Pairwise Potential ($\psi_p$):** Pixels $i$ and $j$ should have same label if they are close (spatial) and have similar color (appearance).
*   **Result:** Snaps segmentation boundaries to image edges.
*   **Why removed?** DeepLab v3 learned to do this internally.

## 2. PSPNet (Pyramid Scene Parsing Network)
Alternative to DeepLab.
*   **Pyramid Pooling Module:**
    *   Pool features at different grid scales: $1 \times 1, 2 \times 2, 3 \times 3, 6 \times 6$.
    *   Upsample and concatenate.
*   **Goal:** Capture global context explicitly. "To know it's a boat, you need to see the water."

## 3. Output Stride
The ratio of input resolution to output resolution.
*   **ResNet:** Stride 32 (Output is $H/32$).
*   **DeepLab:** Uses Atrous Conv in the last two blocks (Stage 4, 5) to remove downsampling.
    *   Output Stride = 16 or 8.
    *   **Trade-off:** Smaller stride = Higher resolution = More computation/memory.

## 4. Fast-SCNN
Real-time segmentation for mobile.
*   **Learning to Downsample:** Efficient initial layers.
*   **Global Feature Extractor:** MobileNet-like block.
*   **Feature Fusion:** Simple addition.
*   Runs at >100 FPS on GPU.

## Summary
While DeepLab focuses on accuracy using Atrous Conv, PSPNet focuses on context using Pyramid Pooling. Both aim to solve the limited receptive field of standard CNNs.

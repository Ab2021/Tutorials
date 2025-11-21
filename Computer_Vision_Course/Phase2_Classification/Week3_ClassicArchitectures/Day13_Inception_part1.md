# Day 13 Deep Dive: Evolution of Inception

## 1. Inception v2 & v3 (2015)
**Factorization:** Decomposing large convolutions.
1.  **$5 \times 5 \to$ Two $3 \times 3$:** Reduces parameters by 28%.
2.  **$N \times N \to 1 \times N + N \times 1$:** Asymmetric factorization.
    *   Replace $3 \times 3$ with $1 \times 3$ followed by $3 \times 1$.
    *   33% savings.
3.  **Label Smoothing:** Introduced here.

## 2. Inception v4 & Inception-ResNet (2016)
**Hybrid:** Combining Inception modules with Residual connections.
*   **Inception-ResNet:** Inception module acts as the residual function $F(x)$.
*   **Scaling:** Residuals are scaled by 0.1 before addition to stabilize training.
*   **Performance:** Faster convergence and slightly better accuracy.

## 3. Depthwise Separable Convolution Analysis
Standard Conv vs Depthwise Separable.
Input: $H \times W \times C_{in}$. Output: $C_{out}$. Kernel: $K \times K$.

**Standard Conv:**
*   Cost: $H \cdot W \cdot C_{in} \cdot C_{out} \cdot K^2$

**Depthwise Separable:**
1.  **Depthwise:** $H \cdot W \cdot C_{in} \cdot K^2$ (1 filter per channel).
2.  **Pointwise:** $H \cdot W \cdot C_{in} \cdot C_{out}$ ($1 \times 1$ conv).
*   **Total Cost:** $H \cdot W \cdot C_{in} \cdot (K^2 + C_{out})$.

**Ratio:**
$$ \frac{\text{DS Conv}}{\text{Std Conv}} = \frac{K^2 + C_{out}}{C_{out} K^2} = \frac{1}{C_{out}} + \frac{1}{K^2} $$
*   For $K=3, C_{out}=128$: Ratio $\approx 1/9$.
*   **9x Faster!**

## 4. Global Average Pooling (GAP)
Replaces the Flatten + Dense layers.
*   **Operation:** Average each feature map to a single number. $7 \times 7 \times 1024 \to 1 \times 1 \times 1024$.
*   **Interpretability:** Each channel corresponds to a specific feature detector (e.g., "dog face detector").
*   **CAM (Class Activation Mapping):** GAP allows visualizing which parts of the image contributed to the class score.

## Summary
The Inception family focused on computational efficiency through factorization. This lineage gave birth to the efficient building blocks (Factorized Convs, Separable Convs) used in mobile architectures today.

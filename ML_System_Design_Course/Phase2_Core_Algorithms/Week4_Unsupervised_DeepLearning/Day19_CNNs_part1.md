# Day 19 (Part 1): Advanced CNN Architectures

> **Phase**: 6 - Deep Dive
> **Topic**: Efficiency & Scale
> **Focus**: MobileNets, Receptive Fields, and Dilation
> **Reading Time**: 60 mins

---

## 1. Efficient Convolutions (MobileNet)

Standard Conv is expensive: $K \times K \times C_{in} \times C_{out}$.

### 1.1 Depthwise Separable Convolution
1.  **Depthwise**: $K \times K \times 1$ filter for *each* input channel. (Spatial correlation).
2.  **Pointwise**: $1 \times 1 \times C_{in} \times C_{out}$. (Channel correlation).
*   **Cost Reduction**: $\frac{1}{C_{out}} + \frac{1}{K^2}$. Roughly 8-9x faster for $3 \times 3$ kernels.

---

## 2. Receptive Field

How much of the image does a neuron see?

### 2.1 The Math
*   $RF_{l} = RF_{l-1} + (K_l - 1) \times \prod_{i=1}^{l-1} S_i$.
*   **Stride ($S$)**: Increases RF exponentially.
*   **Dilation ($D$)**: Increases RF without adding parameters. Effective Kernel size becomes $K + (K-1)(D-1)$.

---

## 3. Tricky Interview Questions

### Q1: What is the purpose of a 1x1 Convolution?
> **Answer**:
> 1.  **Channel Reduction**: Reduce 512 channels to 64 (Bottleneck). Used in Inception/ResNet.
> 2.  **Non-linearity**: Adds ReLU without changing spatial dimensions.
> 3.  **Interaction**: Mixes information across channels.

### Q2: Why does ResNet work? (Beyond "Skip Connections")
> **Answer**:
> *   **Gradient Flow**: The gradient can flow directly through the identity path ($+1$ in derivative). Even if the weight path kills it, the signal survives.
> *   **Ensemble View**: A ResNet with $L$ layers can be seen as an ensemble of $2^L$ shallower networks. Dropping a layer doesn't kill the model.

### Q3: Explain "Aliasing" in CNNs.
> **Answer**:
> *   Downsampling (Strided Conv / Pooling) violates the Nyquist theorem if high frequencies aren't removed.
> *   **Result**: Shift Variance. Moving the input image by 1 pixel changes the prediction completely.
> *   **Fix**: Anti-aliasing (Blur) before downsampling.

---

## 4. Practical Edge Case: Global Average Pooling (GAP)
*   **Old Way**: Flatten -> Dense Layers. Fixed input size required.
*   **New Way**: GAP -> Dense.
*   **Benefit**: Accepts *any* image size. $(N, C, H, W) \to (N, C)$. Drastically reduces parameter count (no dense weights for spatial pixels).


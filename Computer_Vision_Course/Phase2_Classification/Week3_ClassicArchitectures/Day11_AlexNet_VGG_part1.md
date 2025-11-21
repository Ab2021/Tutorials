# Day 11 Deep Dive: Receptive Fields & Design Choices

## 1. The Power of $3 \times 3$ Convolutions
Why did VGG replace $7 \times 7$ with three $3 \times 3$ convs?

1.  **Non-Linearity:** Three layers mean three ReLU activations instead of one. More discriminative power.
2.  **Parameters:**
    *   $7 \times 7 \times C \times C = 49 C^2$
    *   $3 \times (3 \times 3 \times C \times C) = 27 C^2$
    *   **Result:** 45% fewer parameters.
3.  **Receptive Field:**
    *   Layer 1 sees $3 \times 3$.
    *   Layer 2 sees $3 \times 3$ of Layer 1 $\to$ effectively $5 \times 5$ of input.
    *   Layer 3 sees $3 \times 3$ of Layer 2 $\to$ effectively $7 \times 7$ of input.

## 2. Receptive Field Calculation
The region of the input image that affects a particular feature.
$$ RF_{l} = RF_{l-1} + (k_l - 1) \times \prod_{i=1}^{l-1} s_i $$
*   $k_l$: Kernel size at layer $l$.
*   $s_i$: Stride at layer $i$.

**Example:**
*   L1: $3 \times 3$, stride 1. $RF = 1 + (3-1) \times 1 = 3$.
*   L2: $3 \times 3$, stride 1. $RF = 3 + (3-1) \times 1 = 5$.
*   L3: MaxPool $2 \times 2$, stride 2. $RF = 5 + (2-1) \times 1 = 6$ (Wait, pooling affects stride for next layers).

## 3. The VGG Parameter Problem
VGG-16 has ~138 Million parameters.
*   **Conv Layers:** ~14M parameters (10%).
*   **FC Layers:** ~124M parameters (90%).
    *   First FC: $512 \times 7 \times 7 \to 4096$.
    *   Weights: $25,088 \times 4096 \approx 102M$.
*   **Lesson:** Dense layers are heavy. Modern architectures (ResNet, GoogLeNet) replace them with Global Average Pooling to reduce size.

## 4. Local Response Normalization (LRN)
Used in AlexNet, inspired by lateral inhibition in neurobiology.
*   Excited neurons inhibit neighbors.
*   **Formula:**
    $$ b^i_{x,y} = a^i_{x,y} / \left(k + \alpha \sum_{j=\max(0, i-n/2)}^{\min(N-1, i+n/2)} (a^j_{x,y})^2\right)^\beta $$
*   **Verdict:** VGG paper showed it doesn't help much. Batch Normalization replaced it completely.

## 5. Training Tricks of the Time
*   **PCA Color Augmentation:** Altering RGB intensities based on PCA of pixel values (AlexNet).
*   **Multi-Scale Training:** Training on images resized to different scales ($S \in [256, 512]$) (VGG).

# Day 14 Interview Questions: Efficient Architectures

## Q1: What is the difference between a Residual Block and an Inverted Residual Block?
**Answer:**
*   **Residual Block (ResNet):** Wide (High dim) $\to$ Narrow (Low dim bottleneck) $\to$ Wide. Connects high-dim features.
*   **Inverted Residual (MobileNetV2):** Narrow (Low dim) $\to$ Wide (High dim expansion) $\to$ Narrow. Connects low-dim bottlenecks.
    *   **Why?** Depthwise convolution works better on high-dimensional features (more channels = more filters). Memory efficiency is better by keeping the skip connection low-dimensional.

## Q2: Why remove ReLU from the linear bottleneck in MobileNetV2?
**Answer:**
*   ReLU destroys information in the negative region (sets to 0).
*   In high-dimensional space, this information loss is negligible.
*   In low-dimensional space (the bottleneck), ReLU can destroy too much information, causing a "manifold collapse".
*   Using a linear activation preserves all information in the bottleneck.

## Q3: Explain Compound Scaling in EfficientNet.
**Answer:**
A method to scale up a baseline network (B0) to larger versions (B1-B7).
*   Instead of arbitrarily increasing just depth or width, it scales **Depth ($d$), Width ($w$), and Resolution ($r$)** simultaneously using a fixed coefficient $\phi$.
*   $d = \alpha^\phi, w = \beta^\phi, r = \gamma^\phi$.
*   Ensures the network has enough capacity (width/depth) to handle the increased information from higher resolution.

## Q4: What is Hard-Swish and why use it?
**Answer:**
*   **Swish:** $x \cdot \sigma(x)$. Better accuracy than ReLU but expensive (Sigmoid).
*   **Hard-Swish:** $x \cdot \frac{\text{ReLU6}(x+3)}{6}$.
    *   Piece-wise linear approximation.
    *   Much faster to compute on mobile hardware (no exponential function).
    *   Used in MobileNetV3.

## Q5: How does Neural Architecture Search (NAS) work?
**Answer:**
It treats architecture design as an optimization problem.
1.  **Search Space:** Define possible operations (conv $3 \times 3$, $5 \times 5$, skip, etc.).
2.  **Controller:** An RNN or evolutionary algorithm samples an architecture.
3.  **Evaluation:** Train the child network and get accuracy (Reward).
4.  **Update:** Controller learns from reward to sample better architectures.

## Q6: Calculate the FLOPs of a standard vs depthwise separable conv.
**Answer:**
Input $112 \times 112 \times 64$. Output 128 channels. Kernel $3 \times 3$.
*   **Standard:** $112 \times 112 \times 64 \times 128 \times 3 \times 3 \approx 924 \text{ MFLOPs}$.
*   **Depthwise:** $112 \times 112 \times 64 \times 3 \times 3 \approx 7.2 \text{ MFLOPs}$.
*   **Pointwise:** $112 \times 112 \times 64 \times 128 \approx 102 \text{ MFLOPs}$.
*   **Total:** $109.2 \text{ MFLOPs}$ (~8.5x reduction).

## Q7: What is the "Width Multiplier" in MobileNet?
**Answer:**
A hyperparameter $\alpha \in (0, 1]$ that thins the network uniformly.
*   The number of channels at each layer becomes $\alpha \times C$.
*   Computation drops by $\alpha^2$ (quadratic reduction).
*   Allows trading off accuracy for speed/size without redesigning the architecture.

## Q8: Why are $1 \times 1$ convolutions used in MobileNets?
**Answer:**
To perform **Pointwise Convolution**.
*   Depthwise convolution only filters spatially (per channel) and cannot combine features across channels.
*   $1 \times 1$ convs are needed to mix the channels and create new features.

## Q9: What is the Squeeze-and-Excitation (SE) block?
**Answer:**
A channel attention mechanism.
*   **Squeeze:** Global Average Pool to get global context.
*   **Excitation:** A small MLP predicts a weight for each channel.
*   **Scale:** Reweights the channels.
*   Allows the network to emphasize informative channels and suppress useless ones.

## Q10: Implement the Hard-Swish activation.
**Answer:**
```python
class HSwish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6
```

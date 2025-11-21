# Day 16: Attention Mechanisms in Vision

## 1. What is Attention?
**Intuition:** Not all parts of an image are equally important.
*   **Hard Attention:** Cropping a region (non-differentiable).
*   **Soft Attention:** Assigning continuous weights $[0, 1]$ to features (differentiable).
*   **Goal:** Dynamically re-weight features to focus on "what" (Channel) and "where" (Spatial).

## 2. Channel Attention: Squeeze-and-Excitation (SENet, 2017)
**Focus:** "What" is important? (e.g., emphasize "dog ears" channel, suppress "background" channel).

**Mechanism:**
1.  **Squeeze:** Global Average Pooling $\to$ Global context vector ($1 \times 1 \times C$).
2.  **Excitation:** MLP (FC $\to$ ReLU $\to$ FC $\to$ Sigmoid) learns channel weights $w \in \mathbb{R}^C$.
3.  **Scale:** Multiply input feature map $U$ by weights $w$.

```python
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

## 3. Spatial Attention
**Focus:** "Where" is important? (e.g., focus on the center object).

**Mechanism:**
1.  Compress channels (e.g., MaxPool + AvgPool across channels).
2.  Convolution ($7 \times 7$) to produce a spatial map ($H \times W \times 1$).
3.  Sigmoid to get weights.
4.  Multiply input by spatial map.

## 4. CBAM: Convolutional Block Attention Module (2018)
Combines Channel and Spatial attention sequentially.
$$ F' = M_c(F) \otimes F $$
$$ F'' = M_s(F') \otimes F' $$
*   **Channel Module:** MaxPool + AvgPool $\to$ MLP $\to$ Sum $\to$ Sigmoid.
*   **Spatial Module:** MaxPool + AvgPool (channel-wise) $\to$ Conv $\to$ Sigmoid.
*   **Result:** Lightweight module that boosts accuracy on ResNets/MobileNets.

## 5. Self-Attention in CNNs (Non-Local Neural Networks)
**Idea:** Capture long-range dependencies in CNNs (which usually only look at $3 \times 3$).
$$ y_i = \frac{1}{C(x)} \sum_{\forall j} f(x_i, x_j) g(x_j) $$
*   Similar to Transformer attention but applied to feature maps.
*   Computes relationship between pixel $i$ and *every other pixel* $j$.
*   Expensive ($O(N^2)$), so usually applied only at low resolutions.

## Summary
Attention mechanisms allow CNNs to adaptively focus on relevant features, boosting performance with minimal computational overhead (SE/CBAM).

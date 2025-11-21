# Day 14 Deep Dive: Neural Architecture Search (NAS)

## 1. What is NAS?
Automating the design of neural networks.
*   **Search Space:** What architectures can be represented? (e.g., chain of layers, branching, cell-based).
*   **Search Strategy:** How to explore the space? (RL, Evolution, Gradient-based).
*   **Performance Estimation:** How to evaluate a candidate quickly? (Train for few epochs, weight sharing).

## 2. MnasNet (Mobile NAS)
Google's approach that led to MobileNetV3.
*   **Objective:** Maximize Accuracy subject to Latency constraint (on real Pixel phone).
    $$ \text{Reward} = \text{Acc} \times \left( \frac{\text{Latency}}{\text{Target}} \right)^w $$
*   **Result:** Found that $5 \times 5$ convolutions and Squeeze-and-Excitation are useful in specific parts of the network.

## 3. EfficientNet Scaling Analysis
Why Compound Scaling works?
*   **Depth:** Deeper networks capture richer features but are harder to train.
*   **Width:** Wider networks capture fine-grained features but saturate quickly.
*   **Resolution:** Higher resolution sees fine details but requires more receptive field.
*   **Synergy:** Higher resolution needs more depth (larger receptive field) and more width (to capture fine patterns). Scaling them together maintains balance.

## 4. Squeeze-and-Excitation (SE) Block
Lightweight channel attention.
1.  **Squeeze:** Global Average Pooling $\to 1 \times 1 \times C$.
2.  **Excitation:** FC $\to$ ReLU $\to$ FC $\to$ Sigmoid. Learns channel weights $w \in [0, 1]$.
3.  **Scale:** Multiply input channels by weights.
*   Used in MobileNetV3 and EfficientNet.

```python
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Hardsigmoid(inplace=True)
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y
```

## 5. GhostNet (2020)
**Observation:** Feature maps in CNNs often have redundancy (ghosts).
**Idea:**
1.  Generate intrinsic features using cheap convs.
2.  Generate "ghost" features using linear operations (cheap) on intrinsic features.
3.  Concatenate.
*   Even cheaper than MobileNetV2.

## Summary
NAS has shifted architecture design from manual engineering to automated search, resulting in highly optimized families like EfficientNet and RegNet.

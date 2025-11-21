# Day 14: MobileNets & EfficientNets

## 1. MobileNet V1 (2017)
**Goal:** Efficient models for mobile and embedded vision applications.
**Key Innovation:** **Depthwise Separable Convolution** (DS-Conv).
*   Replaces standard convolution with Depthwise + Pointwise.
*   **Hyperparameters:**
    *   **Width Multiplier ($\alpha$):** Scales number of channels (e.g., 0.5, 0.75, 1.0).
    *   **Resolution Multiplier ($\rho$):** Scales input image size.

## 2. MobileNet V2 (2018)
**Key Innovation:** **Inverted Residuals with Linear Bottlenecks**.

**Standard Residual Block (ResNet):**
*   Wide $\to$ Narrow (Bottleneck) $\to$ Wide.
*   ReLU after last conv.

**Inverted Residual Block (MobileNetV2):**
*   Narrow $\to$ Wide (Expansion) $\to$ Narrow.
*   **Expansion:** $1 \times 1$ conv expands channels by factor $t=6$.
*   **Depthwise:** $3 \times 3$ DS-Conv on high-dimensional features.
*   **Projection:** $1 \times 1$ conv projects back to low dimensions.
*   **Linear Bottleneck:** No ReLU after the last projection (preserves information).

```python
class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)
```

## 3. MobileNet V3 (2019)
**Key Innovation:** **Neural Architecture Search (NAS)** + **Squeeze-and-Excitation (SE)**.
*   Used NAS (NetAdapt) to find optimal layer configurations.
*   **Hard Swish:** Faster approximation of Swish activation.
    $$ \text{h-swish}(x) = x \frac{\text{ReLU6}(x+3)}{6} $$
*   **SE Modules:** Lightweight attention added to blocks.

## 4. EfficientNet (2019)
**Problem:** How to scale up a CNN efficiently? (Depth? Width? Resolution?)
**Solution:** **Compound Scaling**.
*   Observation: Scaling dimensions independently is suboptimal.
*   **Method:** Scale all three uniformly with a constant ratio $\phi$.
    *   Depth: $d = \alpha^\phi$
    *   Width: $w = \beta^\phi$
    *   Resolution: $r = \gamma^\phi$
    *   Constraint: $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$ (FLOPS double).

**Architecture:**
*   Base: **EfficientNet-B0** (found via NAS).
*   Scaled: B1 to B7 using compound scaling.
*   Uses **MBConv** (Mobile Inverted Bottleneck) as the building block.

## Summary
MobileNets optimized the building block (DS-Conv, Inverted Residuals) for efficiency. EfficientNet optimized the scaling strategy to maximize accuracy for a given computational budget.

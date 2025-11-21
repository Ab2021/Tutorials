# Day 20: Mobile & Edge Optimization - Theory & Implementation

> **Phase**: 2 - Computer Vision
> **Week**: 4 - Generative & Deployment
> **Topic**: MobileNets, ShuffleNet, and Edge Inference

## 1. Theoretical Foundation: The Efficiency Constraints

Running Deep Learning on a phone/IoT device requires:
1.  **Low Latency**: Real-time (30 FPS = 33ms).
2.  **Low Power**: Cannot drain battery.
3.  **Small Size**: App store limits (e.g., < 100MB).

Standard ResNet-50 (100MB, 4G FLOPs) is too heavy.
We need **Efficient Architectures**.

## 2. MobileNet Family

### MobileNetV1
Introduced **Depthwise Separable Convolution**.
*   Standard Conv: $D_K \times D_K \times M \times N$.
*   Depthwise Sep: $D_K \times D_K \times M$ (Depthwise) + $1 \times 1 \times M \times N$ (Pointwise).
*   Reduces computation by $\approx 8-9x$.

### MobileNetV2
Introduced **Inverted Residuals** and **Linear Bottlenecks**.
*   Expand $\to$ Depthwise $\to$ Project.
*   Skip connections between thin bottlenecks.

### MobileNetV3
Designed via **NAS (Neural Architecture Search)** (NetAdapt).
*   **Hard Swish**: $x \cdot \frac{ReLU6(x+3)}{6}$. Cheaper than Sigmoid/Swish.
*   **Squeeze-and-Excite**: Applied in bottlenecks.

## 3. ShuffleNet
**Grouped Convolutions** are efficient but block information flow between groups.
**Channel Shuffle**:
1.  Group Conv.
2.  Reshape $(G, C/G) \to (C/G, G)$.
3.  Transpose $\to (G, C/G)$.
4.  Flatten.
*   Mixes channels across groups without computation.

## 4. Implementation: Using MobileNetV3

```python
import torch
from torchvision import models

# Load Pre-trained MobileNetV3 Large
model = models.mobilenet_v3_large(weights='DEFAULT')
model.eval()

# Inspect the classifier head
print(model.classifier)
# Sequential(
#   (0): Linear(in_features=960, out_features=1280, bias=True)
#   (1): Hardswish()
#   (2): Dropout(p=0.2, inplace=True)
#   (3): Linear(in_features=1280, out_features=1000, bias=True)
# )

# Export to CoreML (for iOS)
import coremltools as ct

# Trace
example_input = torch.rand(1, 3, 224, 224)
traced_model = torch.jit.trace(model, example_input)

# Convert
model_ct = ct.convert(
    traced_model,
    inputs=[ct.TensorType(shape=example_input.shape)]
)
model_ct.save("MobileNetV3.mlpackage")
```

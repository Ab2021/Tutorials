# Day 13 Interview Questions: Inception & 1x1 Convs

## Q1: What is the primary purpose of a $1 \times 1$ convolution?
**Answer:**
1.  **Dimensionality Reduction:** Reducing the number of channels (depth) to save computation before expensive operations.
2.  **Adding Non-linearity:** Introducing ReLU without changing spatial dimensions.
3.  **Cross-Channel Interaction:** Mixing information from different channels at the same spatial location.

## Q2: Explain the "Inception Hypothesis".
**Answer:**
The hypothesis that optimal local sparse structure in a convolutional vision network can be approximated and covered by readily available dense components.
*   In simple terms: Instead of trying to pick the single best kernel size ($3 \times 3$ or $5 \times 5$), let the network learn to use the best combination of them in parallel.

## Q3: Why did GoogLeNet use Auxiliary Classifiers?
**Answer:**
To combat the **vanishing gradient problem** in deep networks (22 layers was very deep in 2014).
*   They attached small classification heads to intermediate layers.
*   During training, the loss from these heads was added to the total loss (weighted by 0.3).
*   This injected useful gradients directly into the middle of the network.
*   *Note: Inception v3/v4 showed they act more as regularizers.*

## Q4: What is Depthwise Separable Convolution?
**Answer:**
A factorization of standard convolution into two steps:
1.  **Depthwise Conv:** Applies a single filter per input channel (spatial filtering).
2.  **Pointwise Conv:** Applies a $1 \times 1$ convolution to combine the outputs (channel mixing).
*   It is significantly more efficient (computationally and parameter-wise) than standard convolution.

## Q5: How does Global Average Pooling (GAP) prevent overfitting?
**Answer:**
*   Standard FC layers have millions of parameters (e.g., $7 \times 7 \times 512 \times 4096$), which easily overfit.
*   GAP has **zero parameters**. It simply averages the feature maps.
*   It forces the feature maps to be high-level confidence maps for categories, rather than just features for a classifier.

## Q6: Calculate the computational savings of a Depthwise Separable Conv.
**Answer:**
Given $C_{in}=64, C_{out}=128, K=3$.
*   **Standard:** $3 \times 3 \times 64 \times 128 = 73,728$ ops (per pixel).
*   **Depthwise:** $3 \times 3 \times 64 = 576$.
*   **Pointwise:** $1 \times 1 \times 64 \times 128 = 8,192$.
*   **Total Separable:** $576 + 8,192 = 8,768$.
*   **Ratio:** $8,768 / 73,728 \approx 0.118$ (~8.5x reduction).

## Q7: What is the difference between Inception v1 and v3?
**Answer:**
*   **v1 (GoogLeNet):** Standard Inception modules ($1 \times 1, 3 \times 3, 5 \times 5$).
*   **v3:**
    *   Factorized $5 \times 5$ into two $3 \times 3$.
    *   Factorized $N \times N$ into $1 \times N$ and $N \times 1$.
    *   Used RMSProp optimizer.
    *   Introduced Label Smoothing.

## Q8: Why replace $5 \times 5$ convolution with two $3 \times 3$ convolutions?
**Answer:**
1.  **Parameters:** $5 \times 5 = 25$. Two $3 \times 3 = 9 + 9 = 18$. (28% saving).
2.  **Non-linearity:** Two layers allow two activation functions, making the decision function more discriminative.
3.  **Receptive Field:** Same $5 \times 5$ effective field.

## Q9: What is the "Bottleneck" layer in Inception?
**Answer:**
The $1 \times 1$ convolution used *before* the $3 \times 3$ or $5 \times 5$ convolutions.
*   It reduces the number of input channels, making the subsequent expensive convolution cheaper.
*   Example: $256 \xrightarrow{1 \times 1} 64 \xrightarrow{3 \times 3} 64$.

## Q10: Implement a Depthwise Separable Conv in PyTorch.
**Answer:**
```python
class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # Depthwise: groups = in_channels
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                   padding=1, groups=in_channels, bias=False)
        # Pointwise: 1x1
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
    
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

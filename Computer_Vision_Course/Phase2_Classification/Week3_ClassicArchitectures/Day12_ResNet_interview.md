# Day 12 Interview Questions: ResNet

## Q1: Why does adding layers to a plain network (without skip connections) increase training error?
**Answer:**
It's an **optimization problem**, not overfitting.
*   Deep networks suffer from **vanishing/exploding gradients**, making it hard for SGD to converge.
*   Ideally, a deeper model should be able to learn the identity function for extra layers and perform at least as well as a shallower model. The fact that it performs worse means the solver failed to find this solution.

## Q2: How does the Residual connection solve the vanishing gradient problem?
**Answer:**
It creates a **gradient superhighway**.
*   During backpropagation, the gradient of the loss with respect to the input is:
    $$ \frac{\partial L}{\partial x} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial x} = \frac{\partial L}{\partial y} \cdot (1 + \frac{\partial F}{\partial x}) $$
*   The term **1** ensures that the gradient can flow directly back to earlier layers without being multiplied by small weights (which would cause it to vanish).

## Q3: What is the difference between a Basic Block and a Bottleneck Block?
**Answer:**
*   **Basic Block:** Two $3 \times 3$ convs. Used in ResNet-18/34. Good for low channel counts.
*   **Bottleneck Block:** $1 \times 1$ (reduce) $\to 3 \times 3$ (process) $\to 1 \times 1$ (expand). Used in ResNet-50/101/152.
    *   Allows processing high-dimensional features (e.g., 256 channels) efficiently by compressing them first (to 64).

## Q4: Why do we use $1 \times 1$ convolutions in the Bottleneck block?
**Answer:**
To **reduce dimensionality** (and parameters/computation).
*   Instead of doing a $3 \times 3$ conv on 256 channels (expensive), we project to 64 channels using $1 \times 1$, do the $3 \times 3$ conv, and project back to 256.

## Q5: What happens to the dimensions in a skip connection if the input and output sizes don't match?
**Answer:**
We need to match dimensions to perform element-wise addition.
1.  **Spatial mismatch (Stride):** Use a $1 \times 1$ conv with stride 2 (or AvgPool) on the skip connection.
2.  **Channel mismatch:** Use a $1 \times 1$ conv to increase channels.
*   This is called the **projection shortcut**.

## Q6: Compare ResNet and DenseNet.
**Answer:**
*   **ResNet:** Summation ($+$). Features are refined. Gradient flows through identity.
*   **DenseNet:** Concatenation. Features are preserved and reused. Stronger gradient flow but higher memory usage (storing all history).

## Q7: What is "Cardinality" in ResNeXt?
**Answer:**
The number of parallel paths (groups) in a block.
*   ResNeXt splits the channel dimension into $C$ groups (e.g., 32).
*   Increasing cardinality is more effective than increasing depth or width for improving accuracy.

## Q8: Why does ResNet use Global Average Pooling (GAP) at the end?
**Answer:**
*   To reduce the feature map ($7 \times 7 \times 2048$) to a vector ($1 \times 1 \times 2048$).
*   Eliminates the need for massive Fully Connected layers (saving parameters and preventing overfitting).
*   Allows the network to accept images of any size (since GAP always outputs $1 \times 1 \times C$).

## Q9: Can we train a 10,000 layer ResNet?
**Answer:**
Theoretically yes, but practical issues arise (memory, diminishing returns).
*   ResNet-1001 was trained successfully.
*   However, extremely deep networks might just be learning redundant features. Wide ResNets showed that width is often more efficient than extreme depth.

## Q10: Implement a Residual Block with downsampling.
**Answer:**
```python
class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, 1, stride, bias=False),
                nn.BatchNorm2d(out_c)
            )
            
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out
```

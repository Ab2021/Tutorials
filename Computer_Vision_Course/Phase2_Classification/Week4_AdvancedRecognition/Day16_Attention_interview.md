# Day 16 Interview Questions: Attention Mechanisms

## Q1: What is the difference between "Hard" and "Soft" attention?
**Answer:**
*   **Hard Attention:** Makes a discrete decision (e.g., crop a region). It is non-differentiable and requires Reinforcement Learning (REINFORCE) to train.
*   **Soft Attention:** Assigns continuous weights (probabilities) to features. It is differentiable and can be trained end-to-end with Backpropagation.

## Q2: How does Squeeze-and-Excitation (SE) improve feature representation?
**Answer:**
It explicitly models **inter-dependencies between channels**.
*   Standard convolution treats all channels equally (summing them up).
*   SE allows the network to learn that some channels are more important than others for a specific input image (dynamic re-weighting).

## Q3: Why does CBAM use both Max and Avg pooling?
**Answer:**
*   **Avg Pooling:** Good for general background/texture statistics.
*   **Max Pooling:** Good for capturing discriminative features (edges, keypoints).
*   CBAM authors showed that using both provides a more robust feature descriptor than using either alone.

## Q4: What is the computational cost of adding SE blocks?
**Answer:**
**Negligible.**
*   The SE block consists of a Global Pooling and two small FC layers.
*   Compared to the heavy $3 \times 3$ convolutions in the main block, the parameter and FLOP increase is usually < 1-2%, while accuracy gains are significant.

## Q5: Explain the concept of "Non-Local" operations.
**Answer:**
Standard convolution is a **local** operation (looks at $3 \times 3$ neighborhood).
*   To capture long-range dependencies (e.g., relationship between a bird's beak and its tail), a CNN needs many layers to increase the receptive field.
*   **Non-Local** operations (Self-Attention) allow a pixel to attend to *any* other pixel in the image in a single step, capturing global context immediately.

## Q6: Why is dimensionality reduction used in the SE block MLP?
**Answer:**
To reduce parameter count and computation.
*   The first FC layer reduces channels by ratio $r$ (e.g., 16).
*   This creates a bottleneck, forcing the model to learn a compressed representation of channel correlations.
*   It also limits the number of parameters ($C^2/r$ instead of $C^2$).

## Q7: How does Spatial Attention differ from Channel Attention?
**Answer:**
*   **Channel Attention:** "What" to focus on. Reweights feature maps globally (scalar weight per channel).
*   **Spatial Attention:** "Where" to focus on. Reweights pixels spatially (mask per spatial location).

## Q8: Can Attention mechanisms be applied to any CNN?
**Answer:**
**Yes.**
*   Modules like SE, CBAM, and ECA are designed as lightweight blocks that can be inserted after any convolution or residual block in architectures like ResNet, MobileNet, or VGG without changing the overall structure.

## Q9: What is the relationship between Non-Local Networks and Transformers?
**Answer:**
They are mathematically very similar.
*   Non-Local Block is essentially **Self-Attention** applied to computer vision feature maps.
*   The Transformer architecture is built entirely on stacked self-attention blocks.
*   Non-Local Networks usually insert just a few attention blocks into a standard CNN backbone.

## Q10: Implement a Spatial Attention Module.
**Answer:**
```python
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        # Input 2 channels (Max + Avg), Output 1 channel
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg_out = torch.mean(x, dim=1, keepdim=True) # (B, 1, H, W)
        max_out, _ = torch.max(x, dim=1, keepdim=True) # (B, 1, H, W)
        x = torch.cat([avg_out, max_out], dim=1) # (B, 2, H, W)
        x = self.conv(x) # (B, 1, H, W)
        return self.sigmoid(x)
```

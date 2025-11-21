# Day 16 Deep Dive: Advanced Attention & Visualizing

## 1. CBAM Implementation Details
Why MaxPool + AvgPool?
*   **AvgPool:** Captures global statistics (smooth).
*   **MaxPool:** Captures distinctive object features (sharp).
*   Combining both yields richer descriptors than AvgPool alone (used in SE).

```python
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)
```

## 2. ECA-Net (Efficient Channel Attention)
**Critique of SE:** Dimensionality reduction (MLP bottleneck) destroys direct channel correspondence.
**Solution:**
*   No dimensionality reduction.
*   Use 1D Convolution across channels to capture local cross-channel interaction.
*   $k$ neighbors participate in attention for one channel.
*   **Result:** Extremely lightweight and effective.

## 3. BAM (Bottleneck Attention Module)
Similar to CBAM but placed at the bottleneck of the network (between stages) rather than inside every block. Parallel Channel and Spatial branches.

## 4. Visualizing Attention
How do we know it works?
*   **Grad-CAM:** Visualize gradients flowing into the final conv layer.
*   **Attention Maps:** Directly visualize the spatial attention mask $M_s$.
*   **Observation:** Attention models tend to focus more tightly on the target object and ignore background clutter compared to vanilla CNNs.

## Summary
Attention is a modular plug-and-play component. Modern variants like ECA-Net optimize efficiency, while Non-Local Networks bridge the gap to Transformers.

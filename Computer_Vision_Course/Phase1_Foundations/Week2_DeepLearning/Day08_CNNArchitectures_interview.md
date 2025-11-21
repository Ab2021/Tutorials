# Day 8 Interview Questions: CNN Architectures

## Q1: Explain the key innovation of ResNet and why it works.
**Answer:**

**Problem:** Degradation - deeper networks have higher training error.

**Solution:** Residual learning
$$ y = F(x) + x $$

**Why it works:**
1. **Gradient flow:** $\frac{\partial L}{\partial x} = \frac{\partial L}{\partial y}(1 + \frac{\partial F}{\partial x})$ - the "+1" ensures gradients always flow
2. **Identity mapping:** Easy to learn $F(x) = 0$ if needed
3. **Ensemble effect:** Network can be viewed as ensemble of shallow networks

**Empirical results:**
- ResNet-152 (11.3B FLOPs): 77.6% top-1
- Plain-152 (same depth): Worse than Plain-34

## Q2: Compare VGG vs ResNet parameter efficiency.
**Answer:**

**VGG-16:**
- Layers: 16
- Parameters: 138M (mostly in FC layers: 123M)
- FLOPs: 15.5B
- Top-1: 71.5%

**ResNet-50:**
- Layers: 50
- Parameters: 25.6M (5.4× fewer)
- FLOPs: 4.1B (3.8× fewer)
- Top-1: 76.2% (4.7% better)

**Why ResNet is more efficient:**
1. **No large FC layers:** Global average pooling instead
2. **Bottleneck blocks:** 1×1 convs reduce dimensions
3. **Batch normalization:** Better regularization than dropout

**Bottleneck savings:**
- Direct 3×3: $256 \times 256 \times 3 \times 3 = 589,824$ params
- Bottleneck: $256 \times 64 + 64 \times 64 \times 3 \times 3 + 64 \times 256 = 69,632$ params
- **Reduction:** 8.5×

## Q3: What is the bottleneck block and why use it?
**Answer:**

**Structure:**
```
256 channels
    ↓ [1×1, 64]  (reduce)
64 channels
    ↓ [3×3, 64]  (process)
64 channels
    ↓ [1×1, 256] (restore)
256 channels
```

**Benefits:**
1. **Fewer parameters:** 69K vs 590K (8.5× reduction)
2. **Fewer FLOPs:** Computational efficiency
3. **Same receptive field:** 3×3 conv still processes features
4. **More non-linearity:** 3 ReLUs instead of 2

**When to use:**
- Deep networks (ResNet-50+)
- When computational budget is limited
- When you need many layers

## Q4: Explain depthwise separable convolution with FLOPs analysis.
**Answer:**

**Standard convolution:**
- Input: $H \times W \times C_{in}$
- Output: $H \times W \times C_{out}$
- Kernel: $k \times k$
- **FLOPs:** $H \times W \times C_{in} \times C_{out} \times k^2$

**Depthwise separable:**
1. **Depthwise:** $k \times k$ conv per channel
   - FLOPs: $H \times W \times C_{in} \times k^2$
2. **Pointwise:** $1 \times 1$ conv to mix channels
   - FLOPs: $H \times W \times C_{in} \times C_{out}$

**Total FLOPs:** $H \times W \times (C_{in} \times k^2 + C_{in} \times C_{out})$

**Reduction factor:**
$$ \frac{C_{in} \times k^2 + C_{in} \times C_{out}}{C_{in} \times C_{out} \times k^2} = \frac{1}{C_{out}} + \frac{1}{k^2} $$

**Example:** $k=3, C_{out}=256$
$$ \frac{1}{256} + \frac{1}{9} = 0.004 + 0.111 = 0.115 $$
**Reduction:** 8.7×

## Q5: What is DenseNet and how does it differ from ResNet?
**Answer:**

**ResNet:** $y = F(x) + x$ (addition)
**DenseNet:** $y = H([x_0, x_1, ..., x_{l-1}])$ (concatenation)

**Advantages:**
1. **Feature reuse:** All previous features available
2. **Implicit deep supervision:** Short paths to loss
3. **Parameter efficiency:** Fewer parameters for same accuracy
4. **Regularization:** Prevents overfitting

**Growth rate $k$:** Each layer adds $k$ feature maps.

**Parameters:**
- ResNet-50: 25.6M
- DenseNet-201: 20M (fewer, but comparable accuracy)

**Trade-off:** Higher memory usage (concatenation vs addition).

## Q6: Explain EfficientNet's compound scaling.
**Answer:**

**Observation:** Scaling depth, width, or resolution alone is suboptimal.

**Compound scaling:**
$$ d = \alpha^\phi, \quad w = \beta^\phi, \quad r = \gamma^\phi $$

**Constraint:** $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$

**Intuition:**
- Larger images need deeper networks (more layers to process)
- Larger images need wider networks (more channels for details)
- All three should scale together

**Grid search results:** $\alpha=1.2, \beta=1.1, \gamma=1.15$

**Example (B0 → B1):** $\phi=1$
- Depth: $1.2^1 = 1.2$ (20% deeper)
- Width: $1.1^1 = 1.1$ (10% wider)
- Resolution: $1.15^1 = 1.15$ (15% larger)
- FLOPs: $\approx 2^1 = 2×$

**Results:**
- EfficientNet-B0: 5.3M params, 77.1% top-1
- EfficientNet-B7: 66M params, 84.4% top-1 (SOTA at time)

## Q7: Implement a residual block.
**Answer:**

```python
class ResidualBlock(nn.Module):
    """Basic residual block."""
    
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                              stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                              padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.relu = nn.ReLU(inplace=True)
        
        # Shortcut path
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                         stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        identity = self.shortcut(x)
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        out += identity
        out = self.relu(out)
        
        return out
```

## Q8: What is Squeeze-and-Excitation (SE) and why does it help?
**Answer:**

**SE Block:** Channel-wise attention mechanism.

**Steps:**
1. **Squeeze:** Global average pooling
   $$ z_c = \frac{1}{H \times W} \sum_{i=1}^H \sum_{j=1}^W x_c(i,j) $$

2. **Excitation:** Two FC layers with sigmoid
   $$ s = \sigma(W_2 \delta(W_1 z)) $$
   where $\delta$ is ReLU, $W_1 \in \mathbb{R}^{C/r \times C}$, $W_2 \in \mathbb{R}^{C \times C/r}$

3. **Scale:** Element-wise multiplication
   $$ \tilde{x}_c = s_c \cdot x_c $$

**Why it works:**
- **Adaptive recalibration:** Learn channel importance
- **Global context:** Pooling provides global information
- **Minimal overhead:** <1% additional parameters

**Results:** SE-ResNet-50 improves ResNet-50 by ~1% with only 10% more computation.

## Q9: Compare AlexNet, VGG, and ResNet design philosophies.
**Answer:**

**AlexNet (2012):**
- **Philosophy:** Bigger is better, use GPUs
- **Design:** Large filters (11×11, 5×5), decreasing sizes
- **Innovation:** ReLU, dropout, data augmentation
- **Limitation:** Ad-hoc design

**VGG (2014):**
- **Philosophy:** Depth + small filters
- **Design:** Only 3×3 convs, uniform architecture
- **Innovation:** Showed depth matters, simple design
- **Limitation:** Too many parameters (138M), slow

**ResNet (2015):**
- **Philosophy:** Very deep with skip connections
- **Design:** Residual blocks, bottlenecks, batch norm
- **Innovation:** Solved degradation, enabled 1000+ layers
- **Advantage:** Efficient (25M params), accurate, fast

**Evolution:** Ad-hoc → Systematic → Principled

## Q10: How to choose an architecture for a new task?
**Answer:**

**Decision factors:**

1. **Accuracy priority:**
   - Use: EfficientNet-B7, ResNet-152, ViT-Large
   - Trade-off: Slow inference, large model

2. **Speed priority:**
   - Use: MobileNetV3, EfficientNet-B0, ResNet-18
   - Trade-off: Lower accuracy

3. **Mobile/Edge:**
   - Use: MobileNetV2/V3, EfficientNet-Lite
   - Consider: Quantization, pruning

4. **Transfer learning:**
   - Use: ResNet-50, EfficientNet-B0 (good pre-trained weights)
   - Fine-tune: Last layers first, then full network

5. **Custom dataset size:**
   - Small (<10K): MobileNet, EfficientNet-B0 (less overfitting)
   - Large (>100K): ResNet-50, EfficientNet-B3+

**Practical workflow:**
1. Start with ResNet-50 or EfficientNet-B0 (good baseline)
2. Evaluate speed/accuracy trade-off
3. Scale up/down based on requirements
4. Consider domain-specific architectures (medical: U-Net, etc.)

**Example:**
- **ImageNet classification:** EfficientNet-B3
- **Real-time mobile app:** MobileNetV3-Small
- **Medical imaging:** ResNet-50 + transfer learning
- **Object detection:** ResNet-50-FPN backbone

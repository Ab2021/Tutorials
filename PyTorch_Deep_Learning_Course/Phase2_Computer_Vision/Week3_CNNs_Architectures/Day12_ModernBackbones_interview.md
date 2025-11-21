# Day 12: Modern Backbones - Interview Questions

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: ResNet, EfficientNet, and Architecture Design

### 1. Why does ResNet work?
**Answer:**
*   **Gradient Flow**: The identity skip connection allows gradients to flow backwards unimpeded (derivative of $x$ is 1).
*   **Residual Learning**: It's easier to learn a perturbation $F(x)$ towards zero than to learn an identity mapping from scratch.
*   **Loss Landscape**: ResNets have much smoother loss landscapes than plain networks (VGG).

### 2. What is the difference between ResNet-18/34 and ResNet-50/101/152?
**Answer:**
*   **Basic Block**: Used in 18/34. Two $3 \times 3$ convs.
*   **Bottleneck Block**: Used in 50+. $1 \times 1$ (reduce) $\to 3 \times 3 \to 1 \times 1$ (expand).
*   Bottleneck is more parameter efficient for deep networks.

### 3. Explain the "Inverted Residual" block.
**Answer:**
*   Used in MobileNetV2/EfficientNet.
*   Expands channels first (High Dim), does Depthwise Conv, then projects back (Low Dim).
*   Opposite of ResNet Bottleneck (Low $\to$ High $\to$ Low).
*   Memory efficient because the heavy depthwise conv happens in high dim but has few params.

### 4. What is "Compound Scaling" in EfficientNet?
**Answer:**
*   Scaling Depth, Width, and Resolution simultaneously using a principled coefficient $\phi$.
*   Prevents bottlenecks where one dimension limits performance (e.g., huge resolution but shallow network).

### 5. Why did ConvNeXt switch to LayerNorm?
**Answer:**
*   To mimic Transformers.
*   BatchNorm introduces dependencies between samples (batch statistics), which can be problematic.
*   LayerNorm is independent of batch size and works well with the large variations in activation statistics found in modern architectures.

### 6. What is the "Squeeze-and-Excitation" block?
**Answer:**
*   A module that adaptively recalibrates channel-wise feature responses.
*   "Squeeze" global spatial info (AvgPool).
*   "Excite" learns a weight vector (Sigmoid) to scale channels.
*   Allows the network to say "Pay attention to the 'Dog' channel, ignore the 'Background' channel".

### 7. Why use $1 \times 1$ convolutions in bottlenecks?
**Answer:**
*   To reduce dimensionality before the expensive $3 \times 3$ convolution.
*   Example: Reduce 256 channels to 64. $3 \times 3$ on 64 channels is $16\times$ cheaper than on 256.

### 8. What is "Linear Bottleneck" in MobileNetV2?
**Answer:**
*   Removing the ReLU after the final $1 \times 1$ projection layer in the block.
*   Reason: ReLU destroys information in low-dimensional manifolds. Keeping it linear preserves information.

### 9. How does "Stochastic Depth" (DropPath) work?
**Answer:**
*   Regularization technique used in deep ResNets/ConvNeXts.
*   Randomly dropping entire residual blocks during training (skipping $F(x)$).
*   Effectively trains an ensemble of shallower networks.

### 10. What is the advantage of Large Kernels ($7 \times 7$) in ConvNeXt?
**Answer:**
*   Increases effective Receptive Field immediately.
*   Reduces the number of layers needed to see the whole object.
*   Mimics the global attention mechanism of Transformers.

### 11. Why is VGG still used in Style Transfer?
**Answer:**
*   VGG features are very robust and disentangled (Texture vs Content).
*   ResNet features are less interpretable for style due to residual connections mixing features.

### 12. What is "Neural Architecture Search" (NAS)?
**Answer:**
*   Using an algorithm (RL or Evolutionary) to discover the optimal architecture (e.g., EfficientNet was found via NAS).
*   Optimizes for Accuracy vs FLOPs/Latency.

### 13. Explain "Group Normalization".
**Answer:**
*   Splits channels into groups (e.g., 32 groups) and normalizes within each group.
*   Independent of batch size (unlike BN).
*   Better than LayerNorm for CNNs because it respects channel independence to some degree.

### 14. What is the "Receptive Field" of a ResNet-50?
**Answer:**
*   It is massive (larger than the input image).
*   This allows the final features to capture global context.

### 15. Why do we remove the Fully Connected layers in modern backbones?
**Answer:**
*   FC layers have massive parameter counts ($4096 \times 4096$).
*   They require fixed input size.
*   Replaced by Global Average Pooling (0 params, any size).

### 16. What is "Feature Pyramid Network" (FPN)?
**Answer:**
*   Extracting features at multiple scales (stages) from the backbone.
*   Combining high-level semantic features with low-level spatial features.
*   Essential for Object Detection (detecting small and large objects).

### 17. How does DenseNet differ from ResNet?
**Answer:**
*   **ResNet**: Summation ($x + F(x)$).
*   **DenseNet**: Concatenation ($[x, F(x)]$).
*   DenseNet reuses features more efficiently but consumes huge memory due to growing channel count.

### 18. What is "SiLU" (Swish) activation?
**Answer:**
*   $x \cdot \sigma(x)$.
*   Smooth, non-monotonic.
*   Used in EfficientNet and YOLO. Works better than ReLU in deep networks.

### 19. Why are odd kernel sizes ($3, 5, 7$) preferred over even ($2, 4$)?
**Answer:**
*   Odd kernels have a central pixel.
*   Symmetric padding is easier (pad same amount on both sides).
*   Even kernels cause phase shifts/aliasing.

### 20. What is the difference between `torchvision.models.resnet50(pretrained=True)` and `weights=...`?
**Answer:**
*   `pretrained=True` is deprecated.
*   `weights=ResNet50_Weights.IMAGENET1K_V2` is the new API.
*   Allows selecting different versions of weights (V1 vs V2 accuracy).

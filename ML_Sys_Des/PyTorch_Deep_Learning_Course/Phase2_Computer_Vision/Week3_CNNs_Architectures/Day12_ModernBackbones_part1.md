# Day 12: Modern Backbones - Deep Dive

> **Phase**: 2 - Computer Vision
> **Week**: 3 - CNNs & Architectures
> **Topic**: Inverted Residuals, SE Blocks, and Compound Scaling

## 1. MobileNetV2 & The Inverted Residual

Standard ResNet Bottleneck:
`Wide -> Narrow (Bottleneck) -> Wide` + Skip.

MobileNetV2 **Inverted Residual**:
`Narrow -> Wide (Expansion) -> Narrow` + Skip.
*   **Expansion**: $1 \times 1$ conv expands channels by factor of 6.
*   **Depthwise**: $3 \times 3$ depthwise conv filters features.
*   **Projection**: $1 \times 1$ conv projects back to low dim.
*   **Linear Bottleneck**: No ReLU after the last projection (preserves info).

This structure (MBConv) is the core of EfficientNet.

## 2. Squeeze-and-Excitation (SE)

"Attention for Channels".
The network learns to weight importance of different channels.

1.  **Squeeze**: Global Average Pooling $\to (1 \times 1 \times C)$.
2.  **Excitation**: MLP reduces dims ($C/r$) then expands back ($C$) + Sigmoid.
3.  **Scale**: Multiply original feature map by these weights.

$$ X_{new} = X \cdot \sigma(W_2 \delta(W_1 AvgPool(X))) $$

## 3. EfficientNet: Compound Scaling

How to scale up a CNN?
1.  **Width**: More channels (WideResNet).
2.  **Depth**: More layers (ResNet-152).
3.  **Resolution**: Larger images ($512 \times 512$).

EfficientNet paper showed that scaling these 3 dimensions **simultaneously** with a fixed ratio $\phi$ is optimal.
$$ D = \alpha^\phi, W = \beta^\phi, R = \gamma^\phi $$

## 4. ConvNeXt: Modernizing the CNN

How to make a CNN perform like a Vision Transformer (ViT)?
1.  **Patchify**: Use $4 \times 4$ stride 4 conv (like ViT patch embedding).
2.  **Large Kernel**: Use $7 \times 7$ depthwise conv (mimics Attention's global view).
3.  **Inverted Bottleneck**: Similar to Transformer MLP block (expansion ratio 4).
4.  **Fewer Norms/Activations**: Only one LayerNorm and GELU per block.

Result: ConvNeXt beats Swin Transformer without the complexity of Self-Attention.

## 5. Normalization: Batch vs Layer vs Group

*   **BatchNorm**: Normalizes across Batch $(N, H, W)$. Fast, but fails with small batch size.
*   **LayerNorm**: Normalizes across Channels $(C, H, W)$. Independent of batch size. Used in Transformers and ConvNeXt.
*   **GroupNorm**: Normalizes across groups of channels. Robust alternative to BN for detection/segmentation.

# Day 12 Deep Dive: ResNet Variants and Analysis

## 1. Wide ResNet
**Hypothesis:** Depth isn't the only way to increase capacity. Width (number of channels) matters too.
*   **Problem with Deep:** Diminishing returns, slow training (sequential).
*   **Solution:** Decrease depth, increase width ($k \times$ filters).
*   **Result:** WRN-28-10 (28 layers, 10x width) outperforms ResNet-1001. Faster to train (parallelizable).

## 2. DenseNet (Dense Convolutional Network)
**Idea:** Connect *every* layer to *every* subsequent layer in a block.
$$ x_l = H_l([x_0, x_1, \dots, x_{l-1}]) $$
*   **Concatenation** instead of Addition.
*   **Feature Reuse:** Layers can access collective knowledge.
*   **Growth Rate ($k$):** How many new features each layer adds (e.g., $k=32$).
*   **Pros:** Parameter efficient, strong gradient flow.
*   **Cons:** High memory usage (storing all feature maps).

## 3. Stochastic Depth
**Idea:** Randomly drop entire ResNet blocks during training.
*   Like Dropout, but for layers.
*   **Training:** Network is effectively shallower (faster).
*   **Inference:** Use full depth (ensemble effect).
*   Acts as strong regularization.

## 4. ResNet-D (Bag of Tricks)
Tweaks to the original ResNet architecture (from "Bag of Tricks for Image Classification").
1.  **ResNet-B:** Move stride 2 from $1 \times 1$ conv to $3 \times 3$ conv in bottleneck (prevents information loss).
2.  **ResNet-C:** Replace $7 \times 7$ stem with three $3 \times 3$ convs.
3.  **ResNet-D:** Add $2 \times 2$ AvgPool in skip connection for downsampling blocks (instead of strided $1 \times 1$ conv which loses info).

## 5. Identity Mapping Analysis
Why $y = F(x) + x$? Why not $y = F(x) + \lambda x$ or $y = \sigma(F(x) + x)$?
*   **He et al. (2016):** Analyzed propagation.
*   Pure identity path allows gradients to flow unchanged: $\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_L} (1 + \dots)$.
*   Any scaling ($\lambda$) or gating causes gradients to vanish/explode exponentially with depth.
*   **Conclusion:** Keep the skip connection clean.

## Summary
While ResNet is the standard, variants like Wide ResNet (speed/accuracy balance) and ResNeXt (cardinality) offer improvements. DenseNet offers extreme efficiency but high memory cost.

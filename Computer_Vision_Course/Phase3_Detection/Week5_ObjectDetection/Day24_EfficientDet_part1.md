# Day 24 Deep Dive: BiFPN & Feature Fusion

## 1. Feature Fusion Strategies
How to combine features from different levels (e.g., P3, P4, P5)?
1.  **Sum:** $O = I_1 + I_2$. (Assumes equal importance).
2.  **Concatenation:** $O = [I_1, I_2]$. (Increases channels, expensive).
3.  **Softmax Fusion:** $O = \sum \text{softmax}(w_i) \cdot I_i$. (Expensive to compute softmax).
4.  **Fast Normalized Fusion (BiFPN):**
    $$ O = \sum \frac{w_i}{\epsilon + \sum w_j} \cdot I_i $$
    *   $w_i \ge 0$ (ensured by ReLU).
    *   Mathematically similar to softmax but much faster on GPU.

## 2. BiFPN Architecture Details
*   **Cross-Scale Connections:**
    *   Traditional FPN only passes info Top-Down.
    *   BiFPN passes info Top-Down AND Bottom-Up.
    *   It repeats this block multiple times (e.g., 3 times for D0, 8 times for D7).
*   **Skip Connections:**
    *   If a node has an input and output at the same level, add a direct edge.
    *   Preserves gradient flow.

## 3. Scalable Detection Analysis
Why does scaling work?
*   **Resolution:** Higher res helps small objects.
*   **Depth:** Deeper BiFPN allows more complex feature fusion.
*   **Width:** More channels allow capturing more diverse patterns.
*   **Balance:** Scaling just one dimension (e.g., huge resolution on small backbone) hits diminishing returns.

## 4. YOLOv4/v5 vs EfficientDet
*   **YOLOv4:** Uses CSPDarknet backbone + PANet. Optimized for **Speed** (FPS) on GPU.
*   **EfficientDet:** Optimized for **FLOPs** (Efficiency).
*   **Verdict:** YOLO is often faster in practice (latency) because it uses standard convs which are highly optimized on GPUs, whereas EfficientDet uses Depthwise Separable Convs which can be memory-bound.

## Summary
BiFPN is the heart of EfficientDet. Its weighted fusion mechanism allows the network to learn *which* features (high-res or semantic) are most important for a given input.

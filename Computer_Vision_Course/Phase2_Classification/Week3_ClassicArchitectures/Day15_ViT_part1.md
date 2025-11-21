# Day 15 Deep Dive: Attention & Positional Embeddings

## 1. Self-Attention Mechanism
$$ \text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V $$
*   **Query (Q):** What I am looking for?
*   **Key (K):** What do I contain?
*   **Value (V):** What information do I pass?
*   **Global Receptive Field:** Every pixel attends to every other pixel in the first layer!

## 2. Inductive Bias: CNN vs ViT
*   **CNN:**
    *   **Locality:** Pixels are related to neighbors.
    *   **Translation Equivariance:** Object is same regardless of position.
    *   **Hierarchical:** Low-level features $\to$ High-level.
*   **ViT:**
    *   No locality bias (must learn it).
    *   No translation bias (Positional Embeddings needed).
    *   **Benefit:** Can capture long-range dependencies (e.g., shape) better than CNNs.

## 3. Positional Embeddings Analysis
*   **1D vs 2D:** ViT uses 1D learnable embeddings. Surprisingly, 2D sinusoidal embeddings don't improve performance much. The model learns the 2D grid structure from data!
*   **Interpolation:** When changing image size (fine-tuning), the number of patches changes.
    *   We must bicubic-interpolate the pretrained positional embeddings to the new size.

## 4. BEiT (BERT Pre-Training of Image Transformers)
**Idea:** Masked Image Modeling (like BERT).
1.  Tokenize image into visual tokens (using VQ-VAE).
2.  Mask some patches.
3.  Predict the visual tokens of masked patches.
*   Self-supervised pre-training beats supervised pre-training.

## 5. Hybrid Architectures (Convolution + Transformer)
Combining best of both worlds.
*   **Early layers (CNN):** Extract local features efficiently.
*   **Late layers (Transformer):** Model global relationships.
*   **Example:** LeViT, CoAtNet.

## Summary
Transformers trade inductive bias for flexibility. With enough data/regularization, they learn spatial relationships that CNNs hard-code.

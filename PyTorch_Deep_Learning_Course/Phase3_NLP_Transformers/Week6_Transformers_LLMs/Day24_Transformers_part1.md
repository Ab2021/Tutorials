# Day 24: Transformers - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Complexity, Pre-Norm vs Post-Norm, and FFNs

## 1. Complexity Analysis

*   **Self-Attention**: $O(N^2 \cdot d)$. Quadratic with sequence length.
    *   Why? $N \times N$ attention matrix.
    *   Bad for long documents.
*   **Recurrent**: $O(N \cdot d^2)$. Linear with sequence length.
    *   Hard to parallelize.
*   **Convolution**: $O(k \cdot N \cdot d^2)$.
    *   Limited receptive field.

Transformers are preferred because $d$ (1024) is usually larger than $N$ (512), and they are fully parallelizable.

## 2. Feed-Forward Network (FFN)

The FFN in Transformer is applied **position-wise** (independently to each token).
$$ FFN(x) = \max(0, xW_1 + b_1) W_2 + b_2 $$
*   It expands dimensions ($d \to 4d \to d$).
*   Acts as a "Key-Value Memory" storing factual knowledge.

## 3. Pre-Norm vs Post-Norm

Original Paper used **Post-Norm**:
`LayerNorm(x + Sublayer(x))`
*   Hard to train (gradients can vanish/explode). Requires Warmup.

Modern LLMs (GPT-3, LLaMA) use **Pre-Norm**:
`x + Sublayer(LayerNorm(x))`
*   Much more stable training.
*   Allows training deeper networks without warmup.

## 4. Positional Encodings (Deep Dive)

Why Sinusoidal?
*   For any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.
*   Allows the model to extrapolate to lengths longer than seen during training (theoretically).
*   **Learned Embeddings**: (BERT) Just learn a matrix $N \times d$. Easier, but cannot extrapolate.

## 5. Cross-Attention

In the Decoder:
*   **Query**: Comes from Decoder (previous layer).
*   **Key/Value**: Come from Encoder (output).
*   Allows the decoder to "look at" the source sentence to generate the translation.

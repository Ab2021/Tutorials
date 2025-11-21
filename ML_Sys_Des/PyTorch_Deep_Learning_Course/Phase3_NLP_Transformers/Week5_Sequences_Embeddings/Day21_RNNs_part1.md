# Day 21: RNNs - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 5 - Sequences & Embeddings
> **Topic**: Backpropagation Through Time, Exploding Gradients, and LayerNorm

## 1. Backpropagation Through Time (BPTT)

Standard Backprop: $L \to W$.
BPTT: Unroll the RNN for $T$ steps. The loss $L$ depends on $y_T, y_{T-1}, ...$.
$$ \frac{\partial L}{\partial W} = \sum_{t=1}^T \frac{\partial L_t}{\partial W} $$
The gradient flows back from $t=T$ to $t=1$ through the chain rule:
$$ \frac{\partial h_t}{\partial h_{t-1}} = \text{diag}(\sigma'(...)) W_{hh} $$
Repeated multiplication of $W_{hh}$ causes the instability.

## 2. Truncated BPTT

For very long sequences (e.g., 10k words), we can't store the full computation graph (Memory OOM).
**Truncated BPTT**:
1.  Process 100 steps.
2.  Compute Gradients. Update Weights.
3.  **Detach** the hidden state ($h_{100}$.detach()).
4.  Pass $h_{100}$ as initial state for next chunk.
*   Gradients don't flow beyond 100 steps, but memory persists.

## 3. Exploding Gradients & Clipping

If gradients $> 1$, they grow to Infinity.
**Gradient Clipping**: Rescale the gradient vector if its norm exceeds a threshold.
$$ g \leftarrow g \cdot \frac{\text{max\_norm}}{||g||} $$

```python
# PyTorch Implementation
loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
optimizer.step()
```

## 4. Layer Normalization in RNNs

BatchNorm doesn't work well with RNNs (statistics change per time step).
**LayerNorm**: Normalize across the hidden dimension $H$ for each sample independently.
$$ h_t = \frac{h_t - \mu}{\sigma} \cdot \gamma + \beta $$
Crucial for training deep LSTMs and Transformers.

## 5. Bidirectional RNNs

Standard RNN: Past $\to$ Future.
Bidirectional:
*   Forward Layer: $x_1 \to x_T$.
*   Backward Layer: $x_T \to x_1$.
*   Output: Concatenate $[h_{fwd}; h_{bwd}]$.
*   Allows the model to see the future context (e.g., "The **bank** of the river" vs "The **bank** of America").

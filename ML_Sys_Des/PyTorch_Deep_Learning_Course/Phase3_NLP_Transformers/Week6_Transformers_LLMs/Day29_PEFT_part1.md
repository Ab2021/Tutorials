# Day 29: PEFT - Deep Dive

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Prefix Tuning, P-Tuning, and Soft Prompts

## 1. Prompt Tuning (Soft Prompts)

Hard Prompt: "Translate this: ..." (Discrete tokens).
**Soft Prompt**: Learnable vectors prepended to the input embeddings.
*   Input: $[P_1, P_2, ..., P_k, E_{w1}, E_{w2}, ...]$.
*   Only train $P_i$. Freeze model.
*   The model learns "virtual tokens" that guide generation.

## 2. Prefix Tuning

Prompt Tuning only affects the input layer.
**Prefix Tuning**: Prepends learnable key-value pairs to *every* attention layer.
*   More expressive than Prompt Tuning.
*   Similar to LoRA but modifies activations instead of weights.

## 3. IA3 (Infused Adapter by Inhibiting and Amplifying Inner Activations)

Scales the activations by a learned vector.
$$ h = h \odot l $$
*   Multiplies Key and Value vectors in attention.
*   Multiplies intermediate activation in FFN.
*   Extremely parameter efficient.

## 4. Rank ($r$) and Alpha ($\alpha$) in LoRA

*   **Rank ($r$)**: Dimension of the bottleneck.
    *   Small $r$ (8): Good for simple tasks / style transfer.
    *   Large $r$ (64): Needed for complex reasoning / new knowledge.
*   **Alpha ($\alpha$)**: Scaling factor.
    *   Update = $\Delta W \cdot \frac{\alpha}{r}$.
    *   Usually set $\alpha = 2r$ or $\alpha = r$.
    *   Acts like a learning rate multiplier.

## 5. Catastrophic Forgetting in PEFT

PEFT mitigates forgetting because the base model is frozen.
However, the *adapter* can overfit.
*   **Solution**: Train multiple adapters for different tasks.
*   **Hot-Swapping**: Switch adapters at runtime for different users.

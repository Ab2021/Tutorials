# Day 29: Efficient Fine-Tuning (PEFT, LoRA, QLoRA)
## Core Concepts & Theory

### The Problem: Full Fine-Tuning is Expensive

Fine-tuning a 70B parameter model requires updating 70B weights.
*   **VRAM:** You need ~800GB+ of GPU memory (A100 clusters) to store gradients and optimizer states (AdamW).
*   **Storage:** Each checkpoint is 140GB.
*   **Cost:** Prohibitive for most individuals and startups.

### 1. PEFT (Parameter-Efficient Fine-Tuning)

**Idea:** Freeze the main model weights. Only train a tiny number of extra parameters (adapters).
*   **Result:** You only update <1% of parameters.
*   **VRAM:** Drastically reduced (gradients only for adapters).
*   **Performance:** Comparable to full fine-tuning.

### 2. LoRA (Low-Rank Adaptation)

Proposed by Hu et al. (2021).
**Concept:** Matrix decomposition.
*   Weight update $\Delta W$ is a large matrix.
*   LoRA approximates it as $\Delta W = A \times B$, where $A$ and $B$ are low-rank matrices.
*   *Example:* $W$ is $4096 \times 4096$ (16M params). $A$ is $4096 \times 16$, $B$ is $16 \times 4096$ (130k params).
*   **Inference:** You can merge $A \times B$ back into $W$, so there is **zero latency penalty**.

### 3. QLoRA (Quantized LoRA)

Proposed by Dettmers et al. (2023).
**Idea:** Squeeze the base model into 4-bit to save memory, then train LoRA adapters on top.
*   **4-bit NormalFloat (NF4):** An information-theoretically optimal data type for normal distributions (weights).
*   **Double Quantization:** Quantize the quantization constants to save even more space.
*   **Paged Optimizers:** Offload optimizer states to CPU RAM if GPU runs out.
*   **Impact:** You can fine-tune a 65B model on a single 48GB GPU (A6000).

### 4. Other PEFT Methods

*   **Prompt Tuning:** Train soft prompt tokens (embeddings) at the input.
*   **Prefix Tuning:** Prepend trainable vectors to every attention layer.
*   **IA3:** Scale activations with learned vectors.

### Summary

PEFT/LoRA democratized LLM training. It turned "Fine-tuning" from a Google-scale problem into a "run on your gaming PC" problem.

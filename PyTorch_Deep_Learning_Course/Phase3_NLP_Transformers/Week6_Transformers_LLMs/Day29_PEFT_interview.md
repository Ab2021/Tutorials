# Day 29: PEFT & LoRA - Interview Questions

> **Phase**: 3 - NLP & Transformers
> **Week**: 6 - Transformers & LLMs
> **Topic**: Fine-Tuning, Quantization, and Adaptation

### 1. Why is LoRA more memory efficient than Full Fine-Tuning?
**Answer:**
*   We don't need to store optimizer states (Adam moments) for the 7B parameters.
*   We only store them for the adapter (e.g., 4M params).
*   Gradients are also only computed for the adapter.
*   Reduces VRAM requirement by ~3-4x.

### 2. What is the intuition behind Low-Rank Adaptation?
**Answer:**
*   The "Intrinsic Dimension" hypothesis.
*   Over-parameterized models have a low-dimensional manifold of effective weight updates.
*   We don't need to move all 7B parameters independently; moving them in a constrained low-rank subspace is sufficient.

### 3. What is "QLoRA"?
**Answer:**
*   Quantized LoRA.
*   Freezes the base model in 4-bit precision (NF4).
*   Adds LoRA adapters in FP16/BF16.
*   Backpropagates gradients through the frozen 4-bit weights to update the adapters.

### 4. Explain "Soft Prompts".
**Answer:**
*   Learnable vectors prepended to the input sequence.
*   They act like "instructions" in the continuous embedding space.
*   Unlike discrete text prompts, they can be optimized via gradient descent.

### 5. Can LoRA be merged?
**Answer:**
*   Yes. Since LoRA is linear ($\Delta W = BA$), we can add $BA$ to the original weights $W$.
*   $W_{merged} = W + BA$.
*   Inference becomes as fast as the base model (no extra matrix multiplication).

### 6. What is "Double Quantization" in QLoRA?
**Answer:**
*   Quantizing the quantization constants.
*   The quantization scale factors (32-bit) take up space. We quantize them to 8-bit.
*   Saves ~0.5 bits per parameter.

### 7. What is "NF4" (NormalFloat 4-bit)?
**Answer:**
*   A data type optimized for normally distributed weights (which NN weights usually are).
*   Standard INT4 has evenly spaced bins. NF4 has bins spaced according to the quantiles of a Normal distribution.
*   Lower quantization error.

### 8. How do you choose the Rank $r$ in LoRA?
**Answer:**
*   Empirically.
*   Start with $r=8$ or $16$.
*   If underfitting, increase $r$.
*   Paper suggests $r$ can be very small (1 or 2) for some tasks.

### 9. What modules should we apply LoRA to?
**Answer:**
*   Originally: Query and Value projections ($W_q, W_v$).
*   Now: All linear layers (Q, K, V, O, MLP).
*   Applying to all layers usually yields better performance.

### 10. What is "Prefix Tuning" vs "Prompt Tuning"?
**Answer:**
*   **Prompt Tuning**: Adds vectors only at the input layer.
*   **Prefix Tuning**: Adds vectors at *every* layer (modifies keys/values). More powerful, more params.

### 11. What is "Adapter Fusion"?
**Answer:**
*   Training multiple adapters (e.g., one for Science, one for Math).
*   Learning a "Fusion" layer to combine their outputs dynamically based on the input.

### 12. Why do we freeze the base model?
**Answer:**
*   To preserve the pre-trained knowledge.
*   To save memory (no gradients for base model).

### 13. What is the "Alpha" parameter in LoRA?
**Answer:**
*   Scaling factor.
*   The update is scaled by $\alpha / r$.
*   Allows tuning the magnitude of the update without changing the learning rate.

### 14. Can you use LoRA for ConvNets?
**Answer:**
*   Yes. $1 \times 1$ conv is just matrix multiplication.
*   For $3 \times 3$, we can decompose the kernel, but it's less common.

### 15. What is "Paged Optimizer"?
**Answer:**
*   Feature of `bitsandbytes`.
*   If GPU runs out of memory, it automatically moves optimizer states to CPU RAM (paging).
*   Prevents OOM spikes.

### 16. What is "Offloading"?
**Answer:**
*   Moving parts of the model (layers) to CPU or Disk when not in use.
*   Allows running 70B models on small GPUs (very slow).

### 17. Does LoRA increase inference latency?
**Answer:**
*   If merged: No.
*   If unmerged (dynamic adapter switching): Yes, slightly. Need to compute $Wx + BAx$.

### 18. What is "Gradient Checkpointing"?
**Answer:**
*   Trading Compute for Memory.
*   Don't save all intermediate activations. Recompute them during backward pass.
*   Reduces activation memory by $\sqrt{N}$.

### 19. Why initialize $B=0$ in LoRA?
**Answer:**
*   To ensure that at the start of training, the model behaves exactly like the pre-trained model ($\Delta W = 0$).
*   Smooth transition.

### 20. What is "Few-Shot Parameter Efficient Fine-Tuning"?
**Answer:**
*   Combining In-Context Learning (Few-Shot) with PEFT (IA3/LoRA).
*   Achieves SOTA with very few labeled examples.

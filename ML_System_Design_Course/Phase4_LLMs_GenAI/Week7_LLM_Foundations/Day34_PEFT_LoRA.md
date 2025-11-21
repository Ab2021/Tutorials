# Day 34: Parameter Efficient Fine-Tuning (PEFT)

> **Phase**: 4 - LLMs & GenAI
> **Week**: 7 - LLM Foundations
> **Focus**: Training Giant Models on Small GPUs
> **Reading Time**: 45 mins

---

## 1. The Problem: Full Fine-Tuning

Fine-tuning Llama-3-70B requires updating 70 billion parameters.
*   **VRAM**: Needs ~600GB VRAM (Optimizer states + Gradients). Requires 8x A100s.
*   **Storage**: Each fine-tuned version is 140GB.

---

## 2. The Solution: LoRA (Low-Rank Adaptation)

### 2.1 The Hypothesis
Weight updates $\Delta W$ have a low "intrinsic rank". We don't need to change all parameters freely.

### 2.2 The Mechanism
$$W_{new} = W_{frozen} + \Delta W = W_{frozen} + A \times B$$
*   $W$: $d \times d$ matrix (Frozen).
*   $A$: $d \times r$ matrix (Trainable).
*   $B$: $r \times d$ matrix (Trainable).
*   $r$: Rank (e.g., 8, 16, 64). Very small.
*   **Savings**: We only train $A$ and $B$. Reduces trainable parameters by 99.9%.

### 2.3 QLoRA (Quantized LoRA)
*   **Idea**: Load the Base Model in 4-bit (NF4 format). Add LoRA adapters in 16-bit.
*   **Result**: Fine-tune a 70B model on a single 48GB GPU.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Inference Latency
**Scenario**: You have 1 base model and 50 LoRA adapters for different customers.
**Solution**:
*   **LoRA Merging**: $W_{final} = W_{base} + (A \times B)$. Merge the weights offline. Zero latency overhead.
*   **Multi-LoRA Serving**: Keep $W_{base}$ in GPU memory. Swap small $A, B$ matrices on the fly for each request. (vLLM supports this).

### Challenge 2: Rank Selection
**Scenario**: What $r$ should I use?
**Solution**:
*   Start with $r=16$.
*   If underfitting, increase to 64.
*   Alpha scaling: $\alpha = 2 \times r$.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: Why does LoRA reduce VRAM usage?**
> **Answer**:
> 1.  **Gradients**: We only compute gradients for $A$ and $B$ (small), not $W$ (huge).
> 2.  **Optimizer States**: Adam stores 2 states per parameter. LoRA reduces parameters by 1000x, so optimizer memory vanishes.
> 3.  **Note**: It does *not* reduce the memory needed to store activations (forward pass), unless using Gradient Checkpointing.

**Q2: Can you merge QLoRA adapters?**
> **Answer**: Not directly without loss. The base model is quantized (4-bit). Merging 16-bit LoRA weights into 4-bit base weights causes quantization error. You usually dequantize the base model to 16-bit, merge, and then serve.

**Q3: What is the difference between Soft Prompts (P-Tuning) and LoRA?**
> **Answer**:
> *   **Soft Prompts**: Add trainable vectors to the *input* sequence.
> *   **LoRA**: Adds trainable matrices to the *model weights* (Attention/FFN layers).
> *   LoRA generally outperforms Soft Prompts and is more stable.

---

## 5. Further Reading
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [QLoRA Paper](https://arxiv.org/abs/2305.14314)

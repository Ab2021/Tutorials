# Day 103: Edge AI & Small Language Models (SLMs)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Running LLMs on CPU (Llama.cpp)

**Llama.cpp** is the engine of Edge AI.
*   **GGUF Format:** A binary format optimized for fast loading and mapping.
*   **SIMD:** Uses AVX2/NEON instructions to accelerate matrix math on CPUs.
*   **Offloading:** Put 20 layers on GPU, 10 layers on CPU (if VRAM is tight).

### Knowledge Distillation

How to train a good SLM.
1.  **Teacher:** GPT-4.
2.  **Dataset:** Unlabeled text.
3.  **Process:**
    *   Teacher generates output (Soft Labels).
    *   Student trains to match Teacher's logits (KL Divergence Loss).
    *   Student learns the "dark knowledge" (relationships between incorrect classes).

```python
# Pseudo-code for Distillation Loss
loss = alpha * CrossEntropy(student_logits, true_labels) + \
       (1 - alpha) * KLDiv(student_logits, teacher_logits)
```

### Speculative Decoding (On-Device)

Using a Draft model on device, Verify model in cloud? No.
**Draft:** TinyLlama-1.1B.
**Verify:** Llama-3-8B.
Both running locally. The 1.1B model predicts easy tokens ("The cat sat on the..."), the 8B model verifies.

### Summary

*   **Memory Bandwidth:** The bottleneck on edge. Tokens/sec is limited by how fast you can move weights from RAM to CPU.
*   **Battery:** Running AI drains battery. NPUs are 10x more efficient than GPUs.

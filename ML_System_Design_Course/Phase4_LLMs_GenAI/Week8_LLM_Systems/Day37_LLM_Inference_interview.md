# Day 37: LLM Inference - Interview Questions

> **Topic**: Optimization & Serving
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is the KV Cache? Why is it needed?
**Answer:**
*   Stores Key and Value matrices of past tokens.
*   Prevents re-computing attention for the entire prefix at every generation step.
*   Memory intensive.

### 2. Explain PagedAttention (vLLM).
**Answer:**
*   Manages KV Cache like OS manages Virtual Memory (Pages).
*   Allocates non-contiguous memory blocks.
*   Eliminates fragmentation. Increases batch size and throughput.

### 3. What is Continuous Batching (Orca)?
**Answer:**
*   Traditional batching waits for all sequences to finish.
*   **Continuous**: As soon as one sequence finishes, insert a new one into the batch.
*   Maximizes GPU utilization.

### 4. What is Speculative Decoding?
**Answer:**
*   Use a small "Draft Model" to generate K tokens quickly.
*   Verify them with large "Target Model" in parallel (1 forward pass).
*   Accept valid tokens.
*   Speedup without quality loss.

### 5. What is Quantization (INT8 / INT4 / GPTQ / AWQ)?
**Answer:**
*   Represent weights with lower precision.
*   **GPTQ/AWQ**: Post-training quantization optimized for activation distribution.
*   Reduces VRAM usage (load bigger models) and increases memory bandwidth (faster token generation).

### 6. What is the difference between "Time to First Token" (TTFT) and "Inter-Token Latency" (TPOT)?
**Answer:**
*   **TTFT**: Latency to process prompt and output 1st token. (Compute bound).
*   **TPOT**: Time per subsequent token. (Memory Bandwidth bound).

### 7. Why is LLM inference Memory Bandwidth bound?
**Answer:**
*   Generation is auto-regressive (1 token at a time).
*   We load entire model weights + KV cache to compute 1 token.
*   Arithmetic intensity is low.

### 8. What is "Flash Attention" in inference?
**Answer:**
*   Speeds up the prompt processing phase (Prefill).
*   Less impact on decoding phase (unless sequence is very long).

### 9. How does "Tensor Parallelism" work for LLMs?
**Answer:**
*   Split Weight Matrices (Column/Row) across GPUs.
*   Each GPU computes partial result. All-Reduce.
*   Necessary to fit 70B+ models in memory.

### 10. What is "Model Offloading"?
**Answer:**
*   Keep model in CPU RAM / NVMe. Load layers to GPU on demand.
*   Allows running huge models on consumer hardware. Slow.

### 11. What is "Prefix Caching"?
**Answer:**
*   If multiple prompts share same system prompt / context.
*   Compute KV cache for prefix once. Reuse for all requests.
*   Huge speedup for RAG / Chatbots.

### 12. Explain "Temperature" and "Top-P" (Nucleus) sampling.
**Answer:**
*   **Temp**: Scales logits. High = Flat distribution (Random). Low = Peaked (Deterministic).
*   **Top-P**: Sample from smallest set of tokens whose cumulative prob > P. Dynamic vocabulary truncation.

### 13. What is "Top-K" sampling?
**Answer:**
*   Sample from top K most likely tokens.
*   Hard truncation.

### 14. What is "Beam Search" in LLMs?
**Answer:**
*   Explore K paths.
*   Rarely used in Chat (too repetitive/boring). Used in Translation/Summarization.

### 15. What is "Repetition Penalty"?
**Answer:**
*   Penalize logits of tokens that have already appeared.
*   Prevents loops.

### 16. How do you serve multiple LoRA adapters efficiently?
**Answer:**
*   **LoRAX / Punica**.
*   Batch requests for different adapters together.
*   Apply specific LoRA delta $W + A_i B_i$ to specific request in the batch.

### 17. What is "Guided Generation" (Grammar Constraints)?
**Answer:**
*   Force output to follow a schema (JSON, SQL).
*   Mask out tokens that would violate the grammar at each step.

### 18. What is the bottleneck in RAG inference?
**Answer:**
*   Retrieval latency + Long prompt processing (Prefill).

### 19. How does Context Length affect inference cost?
**Answer:**
*   Linear with KV Cache size.
*   Quadratic with Attention (if not using Flash Attention).

### 20. What is "Stop Token"?
**Answer:**
*   Special token (`</s>`, `<|end|>`) that signals model to stop generating.
*   If model fails to emit it, it rambles until max tokens.

# Day 34: Fine-Tuning & PEFT - Interview Questions

> **Topic**: Model Adaptation
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is Full Fine-Tuning?
**Answer:**
*   Updating **all** parameters of the model on a downstream task.
*   Expensive (Memory = Optimizer States for 100% params).
*   Prone to Catastrophic Forgetting.

### 2. What is PEFT (Parameter-Efficient Fine-Tuning)?
**Answer:**
*   Freezing most parameters. Updating only a small subset (adapters).
*   Reduces memory usage. Comparable performance.

### 3. Explain LoRA (Low-Rank Adaptation).
**Answer:**
*   Injects rank decomposition matrices into linear layers.
*   $W' = W + \Delta W = W + A \cdot B$.
*   $A$ is $d \times r$, $B$ is $r \times d$. $r \ll d$.
*   Only train $A$ and $B$.

### 4. Why is LoRA efficient?
**Answer:**
*   Reduces trainable params by 10,000x.
*   Reduces GPU memory (no optimizer states for main weights).
*   No inference latency (merge $A \cdot B$ into $W$).

### 5. What is QLoRA?
**Answer:**
*   Quantized LoRA.
*   Load base model in 4-bit (NF4).
*   Add LoRA adapters in FP16.
*   Backprop through frozen 4-bit weights.
*   Allows finetuning Llama-65B on single 48GB GPU.

### 6. What is Prompt Tuning?
**Answer:**
*   Prepend learnable continuous vectors (soft prompts) to input.
*   Freeze model. Train only soft prompts.

### 7. What is Prefix Tuning?
**Answer:**
*   Prepend learnable vectors to Keys and Values in **every** attention layer.
*   More expressive than Prompt Tuning.

### 8. What is "Catastrophic Forgetting"? How does PEFT help?
**Answer:**
*   Model forgets pre-training knowledge.
*   PEFT freezes base weights, preserving original knowledge.

### 9. What is Instruction Tuning?
**Answer:**
*   Fine-tuning on dataset of (Instruction, Output) pairs.
*   Teaches model to follow commands, not just complete text.
*   Crucial for Chatbots.

### 10. What is the difference between Pre-training and Fine-tuning data?
**Answer:**
*   **Pre-training**: Massive, noisy, general (Common Crawl). Teaches language/facts.
*   **Fine-tuning**: Small, high-quality, specific. Teaches behavior/format.

### 11. How do you choose Rank `r` in LoRA?
**Answer:**
*   Hyperparameter. Usually 8, 16, 64.
*   Higher $r$ = More capacity, more memory.
*   Often $r=8$ is enough.

### 12. Can you combine multiple LoRA adapters?
**Answer:**
*   Yes. Train Adapter A for SQL, Adapter B for Python.
*   Switch adapters at runtime or merge them (weighted sum).

### 13. What is "Adapter Fusion"?
**Answer:**
*   Learning to combine outputs of multiple adapters.

### 14. What is the memory bottleneck in Fine-Tuning?
**Answer:**
*   **Optimizer States** (Adam maintains 2 states per param).
*   **Gradients**.
*   **Activations** (Batch size).
*   Model Weights are actually the smallest part.

### 15. Explain Gradient Checkpointing.
**Answer:**
*   Trade compute for memory.
*   Don't save all intermediate activations. Re-compute them during backward pass.
*   Saves 5x memory, slows down training by 20%.

### 16. What is "Mixed Precision" training?
**Answer:**
*   FP16/BF16 for operations. FP32 for master weights.
*   Speedup and memory savings.

### 17. What is "P-Tuning"?
**Answer:**
*   Variation of Prompt Tuning with an LSTM/MLP encoder to generate soft prompts.

### 18. How does LoRA affect inference speed?
**Answer:**
*   **Merged**: Zero overhead.
*   **Unmerged**: Small overhead (matrix multiplication).

### 19. What is "Self-Instruct"?
**Answer:**
*   Using a strong model (GPT-4) to generate instruction data for a weak model.
*   Alpaca method.

### 20. When should you use Full Fine-Tuning over LoRA?
**Answer:**
*   When the domain is completely new (e.g., Ancient language, DNA sequences).
*   When base model knowledge is insufficient.

# Day 50: Advanced Model Architectures
## Core Concepts & Theory

### Evolution of LLM Architectures

**Progression:**
- **2017:** Transformer (Attention is All You Need)
- **2018:** GPT, BERT (Encoder-only, Decoder-only)
- **2019:** GPT-2, T5 (Scaling up)
- **2020:** GPT-3 (175B parameters)
- **2021:** Switch Transformer (MoE), Codex
- **2022:** PaLM, LLaMA, Chinchilla
- **2023:** GPT-4, LLaMA 2, Mistral
- **2024:** Gemini, Claude 3, GPT-4 Turbo

### 1. Mixture of Experts (MoE)

**Concept:**
- Multiple "expert" networks.
- Router selects which experts to use per token.
- Only activate subset of parameters.

**Architecture:**
```
Input Token → Router → Select Top-K Experts → Combine Outputs
```

**Benefits:**
- **Sparse Activation:** Only use 10-20% of parameters per token.
- **Scaling:** Can scale to trillions of parameters.
- **Efficiency:** Lower inference cost than dense models.

**Examples:**
- **Switch Transformer:** 1.6T parameters, only 10% active.
- **GLaM:** 1.2T parameters, outperforms GPT-3 with less compute.
- **Mixtral 8x7B:** 8 experts, 2 active per token.

### 2. Long-Context Models

**Challenge:** Standard Transformers have O(N²) attention complexity.

**Solutions:**

**Sparse Attention:**
- **Longformer:** Local + global attention.
- **BigBird:** Random + window + global attention.
- **Complexity:** O(N) instead of O(N²).

**Linear Attention:**
- **Performer:** Kernel-based approximation.
- **RWKV:** Recurrent attention.
- **Complexity:** O(N).

**Sliding Window:**
- **Mistral:** Attend to last 4K tokens only.
- **Complexity:** O(N × W) where W is window size.

**Retrieval-Augmented:**
- **Memorizing Transformers:** Retrieve from external memory.
- **Unlimited context** via retrieval.

### 3. Efficient Architectures

**Multi-Query Attention (MQA):**
- Share K/V across all heads.
- **KV Cache:** 8x smaller.
- **Speed:** 1.5-2x faster inference.
- **Models:** PaLM, Falcon.

**Grouped-Query Attention (GQA):**
- Share K/V within groups.
- **Balance:** Between MQA and standard.
- **Models:** LLaMA 2.

**FlashAttention:**
- Optimized attention kernel.
- **Memory:** O(N) instead of O(N²).
- **Speed:** 2-3x faster.

### 4. Multimodal Architectures

**Vision-Language Models:**
- **CLIP:** Contrastive learning for vision-text.
- **Flamingo:** Few-shot vision-language.
- **GPT-4V:** Vision capabilities in GPT-4.

**Architecture:**
```
Image → Vision Encoder → Projection → LLM → Text Output
```

**Audio-Language Models:**
- **Whisper:** Speech recognition.
- **AudioLM:** Audio generation.

### 5. Retrieval-Enhanced Models

**Concept:** Augment LLM with external knowledge retrieval.

**RETRO (Retrieval-Enhanced Transformer):**
```
Input → Retrieve Relevant Docs → Cross-Attend → Generate
```

**Benefits:**
- **Knowledge Update:** Update retrieval DB without retraining.
- **Factuality:** Ground responses in retrieved docs.
- **Efficiency:** Smaller model with retrieval = larger model performance.

### 6. Model Merging

**Concept:** Combine multiple fine-tuned models.

**Methods:**

**Averaging:**
```python
merged_weights = (model1.weights + model2.weights) / 2
```

**Task Arithmetic:**
```python
merged = base + α(model1 - base) + β(model2 - base)
```

**Model Soup:**
- Average checkpoints from same training run.
- **Benefit:** Better generalization.

### 7. Sparse Models

**Concept:** Most parameters are zero.

**Benefits:**
- **Memory:** Store only non-zero weights.
- **Speed:** Skip zero computations.

**Methods:**
- **Magnitude Pruning:** Remove smallest weights.
- **Structured Pruning:** Remove entire neurons/heads.
- **Lottery Ticket Hypothesis:** Find sparse subnetwork.

### 8. Continual Learning

**Challenge:** Catastrophic forgetting (new data overwrites old knowledge).

**Solutions:**

**Elastic Weight Consolidation (EWC):**
- Penalize changes to important weights.

**Progressive Neural Networks:**
- Add new columns for new tasks.

**Memory Replay:**
- Store examples from old tasks, replay during training.

### 9. Neurosymbolic AI

**Concept:** Combine neural networks with symbolic reasoning.

**Approaches:**

**LLM + Code Execution:**
```
LLM generates Python code → Execute → Return result
```
- **Example:** GPT-4 Code Interpreter.

**LLM + Knowledge Graph:**
```
LLM queries KG → Retrieve facts → Generate response
```

**LLM + Theorem Prover:**
```
LLM generates proof steps → Verify with prover
```

### 10. Emerging Architectures

**State Space Models (SSM):**
- **Mamba:** Linear-time sequence modeling.
- **Benefit:** O(N) complexity, competitive with Transformers.

**Hyena:**
- Subquadratic attention via convolutions.
- **Speed:** Faster than Transformers for long sequences.

**RWKV:**
- Recurrent architecture with Transformer-like performance.
- **Benefit:** O(1) inference cost per token.

### Real-World Examples

**Mixtral 8x7B:**
- 8 experts, 2 active per token.
- **Performance:** Matches GPT-3.5 with 7B active parameters.

**GPT-4:**
- Rumored to use MoE architecture.
- Multimodal (vision + text).

**Claude 3:**
- 200K context window.
- Advanced reasoning capabilities.

### Summary

**Architecture Trends:**
- **Sparse Activation:** MoE for efficient scaling.
- **Long Context:** Sparse/linear attention for >100K tokens.
- **Multimodal:** Vision, audio, video integration.
- **Retrieval:** External knowledge augmentation.
- **Efficiency:** MQA, GQA, FlashAttention.

### Next Steps
In the Deep Dive, we will implement MoE, long-context attention, and model merging with complete code examples.

# eBay DS/ML — GenAI, LLMs & Transformers Deep Dive

> eBay is investing heavily in GenAI. Expect 1-2 questions on this in every DS/ML loop.

---

## 🧬 Part 1: Transformer Architecture (Must Know)

### 1.1 Self-Attention Mechanism

```
Input: X ∈ ℝ^(n×d)  [n tokens, d dimensions]

Q = X × W_Q    (Query matrix)
K = X × W_K    (Key matrix)
V = X × W_V    (Value matrix)

Attention(Q,K,V) = softmax(QK^T / √d_k) × V

Why √d_k?  Prevents dot products from growing too large
           → softmax would saturate → vanishing gradients
```

### 1.2 Multi-Head Attention
```
Instead of one attention, run h parallel attention heads:
  head_i = Attention(Q_i, K_i, V_i)
  MultiHead = Concat(head_1, ..., head_h) × W_O

Why? Each head learns different relationship types
     (syntactic, semantic, positional, etc.)
```

### 1.3 Architecture Variants

| Type | Architecture | Key Models | Best For |
|---|---|---|---|
| **Encoder-only** | Bidirectional self-attention | BERT, RoBERTa, eBERT | Classification, embeddings, NER |
| **Decoder-only** | Causal (left-to-right) attention | GPT, Llama, eBayCoder | Text generation, code generation |
| **Encoder-Decoder** | Cross-attention between enc/dec | T5, BART | Translation, summarization |

### 1.4 Key Components
- **Positional Encoding:** Sinusoidal or learned — adds position info since attention is permutation-invariant
- **Layer Normalization:** Stabilizes training, applied before or after attention
- **Residual Connections:** `output = LayerNorm(x + Attention(x))` — prevents degradation in deep networks
- **Feed-Forward Network:** Two-layer MLP after attention in each transformer block

---

## 🏪 Part 2: eBay's GenAI Stack

### 2.1 eBay's Key AI Systems

| System | What It Does | Technology |
|---|---|---|
| **eBayCoder** | Internal code assistant | Fine-tuned Code Llama |
| **e-Llama** | Custom LLM for eBay tasks | Fine-tuned Llama variants |
| **eBERT** | Semantic understanding for search | BERT pre-trained on eBay data |
| **MicroBERT** | Distilled eBERT for production | Knowledge distillation |
| **Mercury** | Agentic RAG recommendation platform | LLM + vector search + tools |
| **Krylov** | ML training platform (GPU cluster) | PyTorch, distributed training |
| **NuKV** | Embedding storage & retrieval | Cloud-native key-value store |

### 2.2 eBay GenAI Use Cases

1. **Automated Listing Generation**
   - Seller uploads photo → Vision model extracts item details → LLM generates title + description + category
   - Challenge: Hallucination prevention, factual accuracy, SEO optimization

2. **Agentic Shopping Assistant**
   - User asks natural language question → Agent plans search → Retrieves products → Generates recommendation
   - Architecture: RAG + tool use + multi-step reasoning

3. **Search Query Understanding**
   - Query → intent classification + entity extraction + query expansion
   - Uses eBERT embeddings for semantic matching

4. **Internal Knowledge Bots**
   - RAG over Jira, GitHub, Wikis → answers engineering questions
   - Reduces onboarding time for new engineers

---

## 🔗 Part 3: RAG (Retrieval-Augmented Generation)

### 3.1 RAG Architecture

```
User Query
    │
    ▼
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│  Embedding   │────►│ Vector Store │────►│  Top-K Docs  │
│  Model       │     │ (Similarity  │     │  (Context)   │
│  (query→vec) │     │  Search)     │     │              │
└─────────────┘     └──────────────┘     └──────┬───────┘
                                                 │
                                                 ▼
                                        ┌──────────────┐
                    User Query ────────►│     LLM      │
                                        │  (Generate   │
                                        │   Answer)    │
                                        └──────────────┘
```

### 3.2 RAG vs Fine-Tuning — When to Use Each

| Dimension | RAG | Fine-Tuning |
|---|---|---|
| **Knowledge updates** | Easy — update vector store | Hard — retrain model |
| **Factual accuracy** | Higher (grounded in docs) | Lower (can hallucinate) |
| **Cost** | Lower (no GPU training) | Higher (GPU hours) |
| **Latency** | Higher (retrieval + generation) | Lower (single pass) |
| **Style/format** | Default LLM style | Custom style (e.g., eBay tone) |
| **Best for** | Knowledge retrieval, Q&A | Task adaptation, domain language |

**eBay's approach: Use BOTH.** RAG for factual grounding + fine-tuned model for domain-specific style and terminology.

### 3.3 RAG Optimization Techniques

| Technique | What It Does |
|---|---|
| **Chunking strategy** | Split docs by semantic sections, not fixed size |
| **Hybrid search** | Combine keyword (BM25) + vector (dense) search |
| **Re-ranking** | Use cross-encoder to re-score retrieved docs |
| **Query transformation** | Decompose complex queries, add hypothetical answers (HyDE) |
| **Corrective RAG (CRAG)** | LLM evaluates if retrieved docs are relevant before using them |
| **Agentic RAG** | LLM decides which tool/source to query for each sub-question |

### 3.4 RAG Evaluation Metrics

| Metric | What It Measures |
|---|---|
| **Context Relevance** | Are retrieved docs relevant to the query? |
| **Faithfulness** | Is the answer grounded in the retrieved context? |
| **Answer Relevance** | Does the answer address the user's question? |
| **Retrieval Precision@K** | What fraction of top-K docs are relevant? |
| **Retrieval Recall@K** | What fraction of all relevant docs are in top-K? |

---

## 🔧 Part 4: Fine-Tuning Techniques

### 4.1 Full Fine-Tuning vs PEFT

| Method | Parameters Updated | GPU Memory | When to Use |
|---|---|---|---|
| **Full fine-tuning** | All parameters | Very high | Unlimited compute, radical domain shift |
| **LoRA** | Low-rank adapter matrices | Low (~10% of full) | Most production use cases |
| **QLoRA** | LoRA + 4-bit quantized base | Very low | Limited GPU budget |
| **Prefix tuning** | Learned prefix tokens | Low | Simple task adaptation |
| **Prompt tuning** | Soft prompt embeddings | Minimal | When model is frozen |

### 4.2 LoRA (Low-Rank Adaptation) — How It Works
```
Original: Y = W × X          (W is frozen)
LoRA:     Y = W × X + B×A×X  (A,B are small trainable matrices)

W ∈ ℝ^(d×d), but A ∈ ℝ^(d×r) and B ∈ ℝ^(r×d) where r << d
Example: d=4096, r=8 → 98.6% fewer parameters to train
```

### 4.3 Catastrophic Forgetting
- **Problem:** Fine-tuning on domain data makes model "forget" general capabilities
- **Mitigation:**
  1. Low learning rate (1e-5 to 5e-5)
  2. Mix domain data with general data during training
  3. Use LoRA (freezes base weights)
  4. Evaluate on general benchmarks alongside domain metrics

---

## ❓ Part 5: 20 GenAI Interview Questions

### Transformer Fundamentals
1. *"Explain self-attention. Why is it O(n²) in sequence length?"*
2. *"What is the difference between encoder-only and decoder-only transformers?"*
3. *"Why does BERT use bidirectional attention while GPT uses causal attention?"*
4. *"What is positional encoding? Why is it necessary?"*
5. *"Explain knowledge distillation. How did eBay create MicroBERT from eBERT?"*

### RAG Systems
6. *"Design a RAG system for eBay's seller help center."*
7. *"How do you evaluate RAG quality? What metrics would you track?"*
8. *"What is the difference between dense retrieval and sparse retrieval? When to use each?"*
9. *"How would you handle a RAG system that returns irrelevant documents?"*
10. *"What is Corrective RAG (CRAG)? How does it improve answer quality?"*

### Fine-Tuning & Adaptation
11. *"When would you choose fine-tuning over RAG for a product feature?"*
12. *"Explain LoRA. Why is it preferred over full fine-tuning in production?"*
13. *"What is catastrophic forgetting? How do you prevent it?"*
14. *"How would you curate a fine-tuning dataset for eBay's listing description generator?"*
15. *"What is the difference between instruction tuning and task-specific fine-tuning?"*

### Production & Applied
16. *"How do you detect and prevent hallucinations in a customer-facing LLM?"*
    - Grounding via RAG, output validation against structured data, confidence scoring, human-in-the-loop
17. *"How would you optimize LLM inference latency for a real-time application?"*
    - Quantization (INT8/INT4), KV-cache, speculative decoding, batching, model distillation
18. *"What is prompt injection? How do you defend against it?"*
    - Input sanitization, system prompt hardening, output filtering, separate user/system contexts
19. *"Design an LLM-based system to auto-generate eBay listing descriptions from photos."*
    - Multi-modal (CLIP/LLaVa) → structured extraction → LLM generation → quality filter → human review
20. *"How do you A/B test an LLM feature? What unique challenges exist?"*
    - Non-deterministic outputs, user preference subjectivity, latency as a metric, prompt sensitivity

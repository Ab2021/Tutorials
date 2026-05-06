# Foundation Model Fundamentals
## AIP-C01 – Domain 1 Core Concepts

---

## 🧠 What Is a Foundation Model (FM)?

A **Foundation Model** is a large-scale AI model trained on massive, diverse datasets using self-supervised learning. It can be adapted to a wide variety of downstream tasks through prompting, fine-tuning, or retrieval augmentation.

### Key Characteristics

| Characteristic | Description |
|---------------|-------------|
| **Scale** | Billions to trillions of parameters |
| **Versatility** | General-purpose; adaptable to many tasks |
| **Training** | Self-supervised on unlabeled data |
| **Adaptation** | Prompting, RAG, fine-tuning, or RLHF |
| **Emergent Behavior** | Capabilities not explicitly trained for appear at scale |

---

## 🔤 Tokenization & Context Windows

### How LLMs Process Text

```
Input Text → Tokenizer → Token IDs → Embedding Layer → Transformer Layers → Output Tokens
```

### Tokens vs Words
- 1 word ≈ 1–4 tokens (depends on language and model tokenizer)
- Code and special characters use more tokens
- A typical token is ~4 characters in English

### Context Window
The **context window** is the maximum number of tokens the model can process at once (input + output).

| Model | Approx Context Window |
|-------|----------------------|
| Claude 3.5 Sonnet | 200,000 tokens |
| Claude 3 Haiku | 200,000 tokens |
| Amazon Nova Pro | 300,000 tokens |
| Llama 3 70B | 128,000 tokens |
| Mistral Large | 32,000 tokens |

> **Exam Tip:** When a question mentions "managing long conversations without hitting context limits," the answer is **conversation summarization** — NOT increasing the context window (which is expensive) or truncation (which loses data).

---

## ⚙️ How LLMs Generate Text (Sampling)

LLMs are **non-deterministic**. They compute a probability distribution over the vocabulary and sample from it.

### Key Inference Parameters

| Parameter | Effect | Exam Context |
|-----------|--------|-------------|
| **Temperature** | Controls randomness (0=deterministic, 1+=creative) | Low temp for factual tasks |
| **Top-P (Nucleus Sampling)** | Considers tokens until cumulative prob reaches P | Balance diversity/quality |
| **Top-K** | Only considers K most probable tokens | Limits vocabulary at each step |
| **Max Tokens** | Maximum output length | Budget control |
| **Stop Sequences** | Terminates generation at specific strings | Structured output control |

> **Exam Trap:** Response hashing fails for RAG regression testing because **temperature > 0 means outputs vary** even for identical inputs. Use model evaluation jobs instead.

---

## 🏗️ Transformer Architecture (Exam Relevant Points)

```
Input Tokens
     ↓
[Token Embeddings] + [Positional Encoding]
     ↓
[Multi-Head Self-Attention] ← Captures relationships between tokens
     ↓
[Feed-Forward Network] ← Per-token transformation
     ↓
[Repeat N times]
     ↓
[Language Model Head] → Probability distribution over vocabulary
```

### Key Concepts for Exam

- **Attention mechanism**: Allows model to weigh importance of different input tokens
- **KV Cache**: Caches Key-Value pairs to speed up autoregressive generation
- **Positional Encoding**: Enables model to understand token order
- **TTFT (Time To First Token)**: Critical for interactive applications; use streaming + latency-optimized config

---

## 📦 Model Modalities

| Modality | Input | Output | Example Use Case |
|----------|-------|--------|-----------------|
| **Text-to-Text** | Text | Text | Chat, summarization, Q&A |
| **Text-to-Image** | Text | Image | Content generation |
| **Image-to-Text** | Image | Text | Document parsing, VQA |
| **Multimodal** | Text + Image/Video/Audio | Text | Invoice processing, video analysis |
| **Text-to-Embedding** | Text | Vector | Semantic search, RAG |
| **Multimodal-to-Embedding** | Text + Image | Vector | Cross-modal retrieval |

---

## 🌟 Amazon Bedrock Foundation Models Catalog

### AWS-Native Models

| Model | Best For | Key Strength |
|-------|---------|-------------|
| **Amazon Nova Micro** | Ultra-low latency, text only | Fastest, cheapest |
| **Amazon Nova Lite** | Balanced speed/cost, multimodal | Strong vision |
| **Amazon Nova Pro** | Complex reasoning, long context | 300K token window |
| **Amazon Nova Canvas** | Image generation | High-quality images |
| **Amazon Titan Text v2** | Text tasks on AWS ecosystem | Native AWS integration |
| **Amazon Titan Embeddings V2** | Text embedding for RAG | Adjustable dimensions (256/384/512/1024) |
| **Amazon Nova Multimodal Embeddings** | Cross-modal search | Text + image + video unified space |

### Anthropic Claude Models (via Bedrock)

| Model | Best For |
|-------|---------|
| **Claude 3.5 Sonnet** | Complex reasoning, coding, nuanced tasks |
| **Claude 3.5 Haiku** | Fast, cost-effective tasks |
| **Claude 3 Opus** | Most powerful for complex tasks |

### Other Partner Models Available

- **Meta Llama 3** (70B, 8B) – Open-source, strong reasoning
- **Mistral Large/7B** – European data privacy use cases
- **Cohere Command R+** – Multilingual, enterprise RAG
- **Stability AI** – Image generation
- **AI21 Jamba** – Long-context document processing

---

## 🔍 Model Selection Decision Framework

```
Start Here
    │
    ├── Is latency critical (interactive chat)?
    │       → Nova Micro / Claude Haiku
    │
    ├── Is complex reasoning required?
    │       → Claude Sonnet / Nova Pro
    │
    ├── Is multimodal input required (images/video)?
    │       → Nova Lite, Nova Pro, Claude 3.5 Sonnet
    │
    ├── Is it a high-volume offline batch task?
    │       → Batch Inference + cost-optimized model
    │
    ├── Is it a semantic search / RAG use case?
    │       → Amazon Titan Embeddings V2
    │
    └── Is cross-modal search required?
            → Nova Multimodal Embeddings
```

---

## 🔄 Model Customization Techniques

### When to Use Each Method

```
┌─────────────────────────────────────────────────────────────────────┐
│                   Model Customization Spectrum                      │
│                                                                     │
│  Prompting  →  RAG  →  Fine-Tuning  →  Continued Pre-Training      │
│  (Easiest,         (Adapts           (Adapts to new domain)         │
│   cheapest)         to style)                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Method Comparison Table

| Method | Purpose | Data Required | Cost | When to Use |
|--------|---------|---------------|------|-------------|
| **Prompt Engineering** | Change behavior via instructions | None | Low | Default starting point |
| **RAG** | Inject dynamic knowledge | Document corpus | Medium | External knowledge needed |
| **Supervised Fine-Tuning (SFT)** | Teach specific task style/format | Labeled Q-A pairs (JSONL) | High | Consistent structured outputs |
| **Continued Pre-Training** | Inject domain vocabulary | Unlabeled domain text | Very High | Specialized jargon (medical, legal) |
| **Reinforcement Fine-Tuning (RFT)** | Align with quality criteria | Reward function / judge | High | Improve accuracy with auto-grading |
| **Model Distillation** | Create smaller model from larger | Teacher model outputs | Medium | Deploy smaller, cheaper model |
| **Custom Model Import** | Use externally trained model | Pre-trained model weights | Variable | SageMaker-trained models in Bedrock |

### Training Data Format (JSONL for Fine-Tuning)

```json
{"prompt": "Summarize this medical report...", "completion": "Patient presented with..."}
{"prompt": "What is the treatment for...", "completion": "Standard treatment involves..."}
```

---

## 🏛️ Key FM Concepts for Exam

### Hallucination
A **hallucination** occurs when an FM generates confident-sounding but factually incorrect information.

**Causes:**
- Training data limitations
- Lack of grounding in retrieved context
- High temperature settings

**Mitigations:**
- RAG (ground responses in retrieved documents)
- Bedrock Guardrails with grounding checks
- Prompt engineering (chain-of-thought, structured output)
- Model evaluation with faithfulness metrics

### Grounding vs. Hallucination

| State | Description |
|-------|-------------|
| **Grounded** | Response is supported by retrieved context |
| **Hallucinated** | Response contains information NOT in context |
| **Faithful** | Response does not contradict source documents |

### Temperature & Determinism

| Temperature | Behavior | Use Case |
|-------------|---------|---------|
| 0 | Greedy (most probable token always) | Factual Q&A, code generation |
| 0.1–0.5 | Low variation, focused | Summarization, classification |
| 0.7–1.0 | Creative, varied | Creative writing, brainstorming |
| >1 | Chaotic, unpredictable | Generally avoid |

---

## 📝 Practice Questions

**Q1:** A company needs a model for a customer service chatbot that must produce consistent, predictable responses to FAQs. Which approach is MOST appropriate?
- A. High temperature with top-K sampling  
- B. Low temperature with a fixed system prompt  
- C. Model fine-tuning on FAQ pairs  
- D. Continued pre-training on customer data  

**Answer: B** – Low temperature produces consistent responses; a system prompt defines the persona. Fine-tuning is overkill for FAQs.

---

**Q2:** A medical company wants to teach an FM to understand proprietary clinical terminology that does not appear in general training data. Which customization method is MOST appropriate?
- A. Prompt engineering with terminology in the system prompt  
- B. RAG with a terminology glossary  
- C. Continued pre-training on clinical documents  
- D. Supervised fine-tuning on Q-A pairs  

**Answer: C** – Continued pre-training injects new vocabulary into the model's weights. Prompting has token limits; RAG is better for dynamic lookup, not vocabulary internalization.

---

*Next: [02_amazon_bedrock_core.md](./02_amazon_bedrock_core.md)*

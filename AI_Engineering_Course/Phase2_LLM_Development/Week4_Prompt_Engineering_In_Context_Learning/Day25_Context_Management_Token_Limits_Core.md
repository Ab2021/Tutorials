# Day 25: Context Management & Token Limits
## Core Concepts & Theory

### The Finite Canvas

LLMs have a fixed context window (e.g., 4k, 8k, 128k tokens).
- **Input + Output <= Limit.**
- If you exceed it, the model crashes or truncates the input.
- **Cost:** Attention scales quadratically $O(N^2)$ (or linearly with Flash Attention), but memory and latency still grow.

### 1. The "Lost in the Middle" Phenomenon

Research (Liu et al., 2023) shows that LLMs are not equally good at using all parts of their context.
- **U-Shaped Curve:** Accuracy is high at the start (Primacy Bias) and end (Recency Bias) of the prompt.
- **Middle:** Accuracy drops significantly for information buried in the middle of a long context.
- **Implication:** Don't just dump 100 documents into the context. Order matters.

### 2. Context Window Expansion Techniques

**A. RoPE Scaling (Interpolation):**
- To extend a pre-trained 4k model to 16k.
- **Linear Interpolation:** Squash the 16k positions into the 0-4k range. Fine-tune.
- **NTK-Aware Scaling:** A mathematical trick to scale high-frequency components less than low-frequency ones. Better extrapolation.

**B. ALiBi (Attention with Linear Biases):**
- Adds a static penalty to attention scores based on distance.
- Allows extrapolation to longer sequences than seen during training without fine-tuning.

### 3. Context Management Strategies

**A. Truncation:**
- Simple: Keep last $N$ tokens.
- Risk: Losing system instructions or key definitions from the start.

**B. Summarization (Map-Reduce):**
- Summarize old conversation turns into a concise paragraph.
- Keep the summary + last few turns in context.

**C. Sliding Window:**
- Only keep the last $K$ tokens.
- Used in Mistral (4096 window).

### 4. RAG as Infinite Context

Instead of fitting everything in context, we store it in a Vector DB.
- **Retriever:** Selects top-k chunks relevant to the query.
- **Generator:** Sees only the relevant chunks.
- **Benefit:** Decouples knowledge size from context size.

### Summary of Limits

| Model | Context Limit | Mechanism |
| :--- | :--- | :--- |
| **GPT-4** | 128k | RoPE Scaling |
| **Claude 3** | 200k | Proprietary |
| **Gemini 1.5** | 1M+ | Ring Attention? |
| **LLaMA-2** | 4k | RoPE |
| **LLaMA-3** | 8k | RoPE |

### Next Steps
In the Deep Dive, we will implement a "Conversation Buffer Window Memory" and analyze RoPE scaling math.

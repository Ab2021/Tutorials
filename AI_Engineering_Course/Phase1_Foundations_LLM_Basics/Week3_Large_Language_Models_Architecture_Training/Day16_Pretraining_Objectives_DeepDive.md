# Day 16: Pre-training Objectives
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Why CLM (Causal Language Modeling) Won

Theoretically, MLM (BERT) should be better because it sees bidirectional context.
However, CLM (GPT) has dominated the LLM era. Why?

**1. The Generative Gap:**
- MLM is trained to *fill in blanks*. It is not trained to *generate text*.
- To generate with BERT, you have to do awkward Gibbs sampling (mask, predict, unmask, repeat), which is slow and poor quality.
- CLM matches the inference process (generate next token) exactly.

**2. Scale Efficiency:**
- **Dense Signal:** CLM predicts a target for *every* token. (Loss on $x_1, x_2, \dots, x_n$).
- **Sparse Signal:** MLM only predicts 15% of tokens.
- To see the same number of training signals, BERT needs to see ~7x more data passes than GPT.

**3. Zero-Shot Generalization:**
- CLM models naturally learn to complete tasks if prompted correctly ("Translate English to French: [Input] -> [Output]").
- MLM models struggle with this format because they weren't trained to continue sequences.

### 2. Prefix LM (The Hybrid Approach)

**Concept:**
Combine the benefits of bidirectional encoding (for the prompt) and autoregressive decoding (for the answer).

**Mechanism:**
- **Prefix (Prompt):** Bidirectional attention (Full visibility).
- **Target (Completion):** Causal attention (Masked future).

**Used In:** PaLM-2, GLM (General Language Model).
**Benefit:** Better performance on understanding tasks (like RAG) where the context is fixed, while maintaining generative capability.

### 3. UL2 (Unifying Language Learning)

Google's UL2 paper proposed a "Mixture of Denoisers". Instead of picking one objective, train on a mix:
1.  **R-Denoiser (Regular):** Standard Span Corruption (T5 style). Good for fine-tuning.
2.  **S-Denoiser (Sequential):** Prefix LM style. Good for few-shot.
3.  **X-Denoiser (Extreme):** Long span masking. Good for long context.

**Result:** A single model that is good at everything (generation, classification, summarization).

### 4. The "Next Sentence Prediction" (NSP) Fallacy

**Original BERT:**
Trained on (Sentence A, Sentence B) pairs.
Task: Is B the actual next sentence?

**Why it failed (RoBERTa findings):**
- The negative examples were often from different documents.
- The model learned to distinguish *topics* rather than *coherence*.
- It was too easy.
- **Better Approach:** Train on **Full Sentences** (contiguous text from one document) that pack the context window (e.g., 512 tokens). This forces the model to learn long-range dependencies.

### 5. Code Pre-training (Fill-In-The-Middle)

For coding models (Codex, StarCoder), standard CLM is not enough.
Developers often edit code in the middle of a file, not just at the end.

**FIM (Fill-In-The-Middle) Objective:**
- Original: `def add(a, b): return a + b`
- Transformation: Move the middle to the end.
- Input: `<PRE> def add(a, b): <SUF> return a + b <MID>`
- Target: `return a + b`

**Result:**
The model learns to generate code conditioned on both the *prefix* (imports, function signature) and the *suffix* (closing brackets, usage later in file).

### Summary of Mechanics

| Objective | Attention Mask | Loss Calculation | Efficiency |
| :--- | :--- | :--- | :--- |
| **CLM** | Triangular (Causal) | All tokens | High |
| **MLM** | Full (Bidirectional) | Masked tokens (15%) | Low |
| **Prefix LM** | Full (Prefix) + Causal (Target) | Target tokens | Medium |
| **FIM** | Causal (reordered) | All tokens | High |

### Code: Implementing Objectives

```python
# CLM Mask (Standard)
def create_clm_mask(size):
    return torch.tril(torch.ones(size, size))

# Prefix LM Mask
def create_prefix_mask(prefix_len, target_len):
    total = prefix_len + target_len
    mask = torch.zeros(total, total)
    
    # Prefix part: Bidirectional (can see everything in prefix)
    mask[:prefix_len, :prefix_len] = 1
    
    # Target part: Causal (can see prefix + past targets)
    mask[prefix_len:, :prefix_len] = 1 # See prefix
    mask[prefix_len:, prefix_len:] = torch.tril(torch.ones(target_len, target_len))
    
    return mask
```

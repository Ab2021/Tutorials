# Day 21: Evaluation Metrics for Language Models
## Core Concepts & Theory

### The Evaluation Crisis

Evaluating LLMs is harder than training them.
- **Classification:** Accuracy is easy.
- **Generation:** "Write a poem about a cat." -> How do you score this?
There is no single "Gold Standard". We rely on a mix of automated metrics and human evaluation.

### 1. Perplexity (PPL)

**Definition:** The exponentiated average negative log-likelihood of a sequence.
$$ PPL(X) = \exp\left( -\frac{1}{t} \sum_{i=1}^t \log P(x_i | x_{<i}) \right) $$
**Intuition:** How "surprised" is the model by the text?
- Lower is better.
- A PPL of 10 means the model is as confused as if it were choosing uniformly from 10 options at each step.
**Use Case:** Pre-training sanity check. Comparing models on the *same tokenizer*.
**Limitation:** Does not correlate well with generation quality (a model can have low PPL but repeat text).

### 2. N-gram Overlap Metrics (Legacy)

**BLEU (Bilingual Evaluation Understudy):**
- Measures precision of n-grams (1-4) between candidate and reference.
- Standard for Machine Translation.
- **Flaw:** "The cat sat on the mat" vs "The mat was sat on by the cat". Low BLEU, same meaning.

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Measures recall of n-grams.
- Standard for Summarization.
- **ROUGE-L:** Longest Common Subsequence.

### 3. Semantic Metrics (Embedding-based)

**BERTScore:**
- Compute contextual embeddings (BERT) for candidate and reference.
- Calculate cosine similarity between token embeddings.
- **Benefit:** Captures synonyms and paraphrasing.

**Mauve:**
- Compares the distribution of generated text to human text using divergence metrics.
- Good for open-ended generation.

### 4. LLM-as-a-Judge

**Concept:** Use a stronger LLM (GPT-4) to evaluate a weaker LLM (LLaMA-7B).
- **Prompt:** "Please rate the following response on a scale of 1-10 for helpfulness and accuracy..."
- **MT-Bench:** A set of multi-turn questions graded by GPT-4.
- **AlpacaEval:** Head-to-head comparison (Win Rate) against a baseline (Davinci-003).

### 5. Benchmarks

- **MMLU (Massive Multitask Language Understanding):** 57 subjects (Math, History, Law). Multiple choice. Standard for general knowledge.
- **GSM8K:** Grade School Math. Tests multi-step reasoning.
- **HumanEval:** Python coding problems.
- **HellaSwag:** Common sense reasoning.

### Summary of Metrics

| Metric | Type | Best For |
| :--- | :--- | :--- |
| **Perplexity** | Internal | Pre-training progress |
| **BLEU/ROUGE** | Overlap | Translation/Summarization |
| **BERTScore** | Semantic | Paraphrasing |
| **MMLU** | Benchmark | General Knowledge |
| **LLM-as-Judge** | Model-based | Chatbot quality |

### Next Steps
In the Deep Dive, we will analyze why Perplexity fails for Chatbots and how to implement a custom LLM-as-a-Judge pipeline.

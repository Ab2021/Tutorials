# Day 39: Evaluation of LLMs

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: Measuring the Unmeasurable
> **Reading Time**: 45 mins

---

## 1. The Evaluation Crisis

Traditional metrics (Accuracy, F1) don't work for "Write a poem". BLEU/ROUGE (n-gram overlap) are useless for semantic correctness.

### 1.1 LLM-as-a-Judge
*   **Idea**: Use GPT-4 to grade Llama-3's answer.
*   **Prompt**: "You are a judge. Rate this answer from 1-5 based on helpfulness."
*   **Pros**: Scalable, correlates well with human preference.
*   **Cons**: Self-bias (GPT-4 prefers GPT-4 style outputs).

### 1.2 RAG Evaluation (RAGAS)
*   **Faithfulness**: Is the answer supported by the retrieved context?
*   **Answer Relevance**: Does the answer address the user query?
*   **Context Precision**: Is the relevant chunk at the top?

---

## 2. Benchmarks

### 2.1 General
*   **MMLU**: Multiple choice knowledge (Math, History, Law).
*   **Chatbot Arena (LMSYS)**: Elo rating based on blind human pairwise comparisons. The Gold Standard.

### 2.2 Coding
*   **HumanEval**: Python coding problems. Metric: `pass@1` (Does the code pass unit tests?).

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Data Contamination
**Scenario**: Llama-3 scores 90% on MMLU.
**Reason**: MMLU questions were in its training set. It memorized them.
**Solution**:
*   **Decontamination**: Remove benchmark data from training set.
*   **Dynamic Benchmarks**: Private, rotating questions.

### Challenge 2: Cost of Eval
**Scenario**: Running GPT-4-as-a-judge on 10k rows costs $500.
**Solution**:
*   **Sampling**: Eval on random 100 rows.
*   **Small Judge**: Use a fine-tuned Llama-3-70B as a judge instead of GPT-4.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is `pass@k` metric?**
> **Answer**: In coding, we generate $k$ solutions. If *any* of them passes the unit tests, it's a success. `pass@1` is strict. `pass@10` measures the model's ability to eventually get it right.

**Q2: How do you evaluate Hallucination?**
> **Answer**:
> *   **NLI (Natural Language Inference)**: Use a small model (DeBERTa) to check if "Premise entails Hypothesis".
> *   **Fact Checking**: Extract claims -> Search Google -> Verify claims.

**Q3: Why is BLEU bad for LLMs?**
> **Answer**: BLEU measures word overlap.
> *   Ref: "The cat sat on the mat."
> *   Pred: "A feline rested on the rug."
> *   BLEU score is near 0, but the meaning is identical. Semantic similarity (Embeddings) or LLM-Judges are better.

---

## 5. Further Reading
- [RAGAS Documentation](https://docs.ragas.io/en/latest/)
- [Chatbot Arena Leaderboard](https://chat.lmsys.org/)

# Day 39 (Part 1): Advanced LLM Evaluation

> **Phase**: 6 - Deep Dive
> **Topic**: Grading the AI
> **Focus**: RAGAS, LLM-as-a-Judge, and Bias
> **Reading Time**: 60 mins

---

## 1. RAGAS Metrics

How to evaluate RAG without ground truth?

### 1.1 Faithfulness
*   **Goal**: Is the answer derived *only* from the context? (No Hallucination).
*   **Method**: Extract claims from Answer. Verify each claim against Context using an LLM.
*   Score = Supported Claims / Total Claims.

### 1.2 Answer Relevance
*   **Goal**: Does the answer address the query?
*   **Method**: Generate *Question* from the *Answer*. Calculate Cosine Similarity between Generated Question and Original Question.

---

## 2. LLM-as-a-Judge

### 2.1 Pairwise Comparison
*   Prompt: "Compare Answer A and Answer B. Which is better?"
*   **Positional Bias**: LLMs prefer the first option.
*   **Fix**: Swap order and average.

### 2.2 Self-Consistency
*   Ask the judge 5 times with temp > 0. Take majority vote.

---

## 3. Tricky Interview Questions

### Q1: Perplexity vs Task Accuracy?
> **Answer**:
> *   Low Perplexity $\neq$ High Accuracy.
> *   A model can be very confident (Low PP) but wrong.
> *   Always evaluate on downstream task (Exact Match, F1).

### Q2: How to measure Hallucination Rate?
> **Answer**:
> *   **NLI (Natural Language Inference)**: Entailment check.
> *   **FactChecking**: Use Google Search API to verify entities.

### Q3: BLEU/ROUGE for LLMs?
> **Answer**:
> *   **Useless**. They measure n-gram overlap.
> *   "The cat sat" vs "The feline rested". Zero overlap, same meaning.
> *   Use **BERTScore** or **G-Eval**.

---

## 4. Practical Edge Case: Contamination
*   **Problem**: Test set was in the training data (Common Crawl).
*   **Check**: N-gram overlap between Test Set and Pretraining Corpus.
*   **Fix**: Canary Strings (GUIDs) injected into test set to detect leakage.


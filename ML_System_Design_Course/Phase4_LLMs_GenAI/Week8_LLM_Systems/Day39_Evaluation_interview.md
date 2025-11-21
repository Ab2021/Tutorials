# Day 39: Evaluation - Interview Questions

> **Topic**: LLM Metrics
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. How do you evaluate an LLM?
**Answer:**
*   **Benchmarks**: MMLU (Knowledge), GSM8K (Math), HumanEval (Code).
*   **Human Eval**: Chatbot Arena.
*   **Model-based Eval**: LLM-as-a-Judge.

### 2. What is Perplexity?
**Answer:**
*   Measure of how surprised the model is by the text.
*   $PPL = \exp(Loss)$.
*   Good for pre-training check. Bad proxy for generation quality.

### 3. What is BLEU / ROUGE? Are they good for LLMs?
**Answer:**
*   N-gram overlap metrics.
*   **Bad for LLMs**. "The cat sat" vs "Feline rested" -> Low BLEU, but same meaning.
*   Used for Translation/Summarization historically.

### 4. What is "LLM-as-a-Judge"?
**Answer:**
*   Use GPT-4 to grade the output of a smaller model (1-10 score).
*   Correlates well with human judgment.
*   Fast and cheap.

### 5. What is RAGAS (RAG Assessment)?
**Answer:**
*   Framework for evaluating RAG.
*   **Faithfulness**: Is answer derived from context?
*   **Answer Relevance**: Does answer address query?
*   **Context Precision/Recall**: Did retrieval get right docs?

### 6. What is "Hallucination Rate"?
**Answer:**
*   Percentage of generated facts that are false.
*   Measured by Fact-Checking (using Search or Knowledge Graph).

### 7. What is MMLU (Massive Multitask Language Understanding)?
**Answer:**
*   Multiple choice questions across 57 subjects (Math, History, Law).
*   Standard benchmark for general knowledge.

### 8. What is "Needle in a Haystack" test?
**Answer:**
*   Insert a specific fact (Needle) into a long context (Haystack).
*   Ask model to retrieve it.
*   Tests long-context capability.

### 9. How do you evaluate Code Generation?
**Answer:**
*   **Pass@k**: Generate k solutions. Run unit tests. If at least one passes -> Success.
*   HumanEval / MBPP benchmarks.

### 10. What is "Elo Rating" for LLMs?
**Answer:**
*   Comparative ranking.
*   Model A vs Model B. Winner gains points.
*   LMSYS Chatbot Arena.

### 11. What is BERTScore?
**Answer:**
*   Compute similarity of embeddings (BERT) between Reference and Candidate.
*   Better than BLEU (captures semantics).

### 12. What is "Red Teaming"?
**Answer:**
*   Adversarial testing.
*   Trying to break the model (Jailbreak, Toxicity, Bias).

### 13. How do you evaluate Chatbots?
**Answer:**
*   Multi-turn consistency.
*   Persona adherence.
*   User satisfaction (Thumbs up/down).

### 14. What is "Data Contamination" in evaluation?
**Answer:**
*   Test set questions were present in the training data.
*   Model memorized answers.
*   Inflates scores.

### 15. What is "Length Bias" in LLM-as-a-Judge?
**Answer:**
*   Judges (GPT-4) tend to prefer longer answers, even if they are verbose/worse.

### 16. What is "Self-Consistency"?
**Answer:**
*   Generate multiple answers. Take majority vote.
*   Improves performance. Can also be used as a confidence metric.

### 17. How do you evaluate Latency/Throughput?
**Answer:**
*   **TTFT** (Time to First Token).
*   **Tokens Per Second** (TPS).
*   Load testing.

### 18. What is "G-Eval"?
**Answer:**
*   Framework using LLMs with Chain-of-Thought to evaluate outputs based on custom criteria (Coherence, Engagingness).

### 19. How do you evaluate Safety?
**Answer:**
*   Benchmarks like **Do Not Answer**.
*   Attack success rate.

### 20. Why is evaluation hard?
**Answer:**
*   Open-ended generation has no single "Correct" answer.
*   Subjective.
*   Benchmarks saturate quickly.

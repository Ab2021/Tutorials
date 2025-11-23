# Day 56: Evaluation & Benchmarking
## Core Concepts & Theory

### Evaluation Fundamentals

**Challenge:** LLM outputs are open-ended and subjective

**Evaluation Types:**
- Automatic metrics
- Human evaluation
- LLM-as-judge
- Benchmark datasets

### 1. Automatic Metrics

**Perplexity:**
- Measure of uncertainty
- Lower = better language modeling
- **Formula:** `exp(average_cross_entropy_loss)`

**BLEU (Bilingual Evaluation Understudy):**
- N-gram overlap with reference
- **Range:** 0-100 (higher better)
- **Use Case:** Machine translation

**ROUGE (Recall-Oriented Understudy for Gisting Evaluation):**
- Recall-based n-gram overlap
- **ROUGE-1:** Unigram overlap
- **ROUGE-L:** Longest common subsequence
- **Use Case:** Summarization

**BERTScore:**
- Semantic similarity using BERT embeddings
- **Range:** 0-1 (higher better)
- **Benefit:** Captures meaning, not just surface form

### 2. Task-Specific Benchmarks

**MMLU (Massive Multitask Language Understanding):**
- 57 subjects (STEM, humanities, social sciences)
- Multiple choice questions
- **Metric:** Accuracy

**HellaSwag:**
- Commonsense reasoning
- Sentence completion
- **Metric:** Accuracy

**TruthfulQA:**
- Factual accuracy
- Tests for hallucinations
- **Metric:** % truthful answers

**HumanEval:**
- Code generation
- 164 programming problems
- **Metric:** pass@k (% passing unit tests)

### 3. Conversational Benchmarks

**MT-Bench:**
- Multi-turn conversations
- 80 questions across 8 categories
- **Metric:** GPT-4 judge score (1-10)

**AlpacaEval:**
- Single-turn instructions
- 805 questions
- **Metric:** Win rate vs reference model

**Chatbot Arena:**
- Human preference via Elo ratings
- Head-to-head comparisons
- **Metric:** Elo score

### 4. LLM-as-Judge

**Concept:** Use strong LLM (GPT-4) to evaluate other LLMs

**Prompt Template:**
```
Evaluate the following response on a scale of 1-10:

Question: {question}
Response: {response}

Criteria:
- Helpfulness
- Accuracy
- Clarity

Score:
```

**Benefits:**
- Scalable
- Consistent
- Correlates with human judgment

**Limitations:**
- Bias towards certain styles
- Can't evaluate factual accuracy perfectly

### 5. Human Evaluation

**Methods:**

**Pairwise Comparison:**
- Show two responses
- Ask which is better
- **Metric:** Win rate

**Likert Scale:**
- Rate response 1-5
- **Dimensions:** Helpfulness, accuracy, safety

**Task Success:**
- Did response accomplish task?
- **Metric:** Success rate

### 6. Safety Evaluation

**Toxicity:**
- Perspective API score
- **Target:** <0.1

**Bias:**
- BBQ (Bias Benchmark for QA)
- **Metric:** Bias score

**Jailbreak Resistance:**
- Red teaming attempts
- **Metric:** Refusal rate

### 7. Efficiency Metrics

**Latency:**
- **TTFT (Time to First Token):** <500ms
- **TPOT (Time Per Output Token):** <50ms
- **Total Latency:** p95 <2s

**Throughput:**
- Requests/second
- Tokens/second

**Cost:**
- $ per 1K tokens
- $ per request

### 8. Calibration

**Concept:** Model's confidence matches accuracy

**Measurement:**
- Expected Calibration Error (ECE)
- **Formula:** `Σ |accuracy - confidence| × bin_weight`

**Well-calibrated:** 80% confidence → 80% accuracy

### 9. Real-World Benchmarks

**GPT-4 Scores:**
- MMLU: 86.4%
- HumanEval: 67.0%
- TruthfulQA: 59%

**Claude 3 Opus:**
- MMLU: 86.8%
- HumanEval: 84.9%
- MT-Bench: 9.0/10

**LLaMA 2 70B:**
- MMLU: 68.9%
- HumanEval: 29.9%

### 10. Evaluation Best Practices

**Multiple Metrics:**
- Don't rely on single metric
- Use task-specific + general benchmarks

**Human Evaluation:**
- Sample 100-500 examples
- Multiple annotators per example

**Continuous Evaluation:**
- Evaluate every model version
- Track metrics over time

**Domain-Specific:**
- Create custom benchmarks for your domain
- Measure what matters for your use case

### Summary

**Evaluation Strategy:**
1. **Automatic Metrics:** BLEU, ROUGE, BERTScore (quick feedback)
2. **Benchmarks:** MMLU, HumanEval, MT-Bench (standardized comparison)
3. **LLM-as-Judge:** GPT-4 evaluation (scalable quality assessment)
4. **Human Evaluation:** Pairwise comparison (ground truth)
5. **Safety:** Toxicity, bias, jailbreak resistance
6. **Efficiency:** Latency, throughput, cost

### Next Steps
In the Deep Dive, we will implement complete evaluation pipeline with automatic metrics, LLM-as-judge, and human evaluation.

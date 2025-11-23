# Day 56: Evaluation & Benchmarking
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: What is the difference between BLEU and ROUGE?

**Answer:**
**BLEU:**
- Precision-based (how much of prediction is in reference)
- N-gram overlap
- **Use Case:** Machine translation
- **Range:** 0-100

**ROUGE:**
- Recall-based (how much of reference is in prediction)
- **ROUGE-1:** Unigram overlap
- **ROUGE-L:** Longest common subsequence
- **Use Case:** Summarization

**Key Difference:** BLEU penalizes missing words, ROUGE penalizes extra words.

#### Q2: How does LLM-as-judge work?

**Answer:**
1. **Create evaluation prompt** with question, response, criteria
2. **Send to strong LLM** (GPT-4) for judgment
3. **Parse scores** from response
4. **Aggregate** across examples

**Benefits:** Scalable, consistent, correlates with human judgment
**Limitations:** Bias towards certain styles, can't verify facts perfectly

#### Q3: What is MT-Bench and how is it scored?

**Answer:**
- **Multi-turn conversations:** 80 questions, 2 turns each
- **8 categories:** Writing, roleplay, reasoning, math, coding, extraction, STEM, humanities
- **Scoring:** GPT-4 judges each turn on 1-10 scale
- **Final score:** Average across all turns and categories

**Example:** GPT-4 scores 9.0/10, Claude 3 Opus scores 9.0/10

#### Q4: What is calibration and why is it important?

**Answer:**
- **Calibration:** Model's confidence matches accuracy
- **Well-calibrated:** 80% confidence → 80% correct
- **Measurement:** Expected Calibration Error (ECE)
- **Importance:** Users need to trust confidence scores for decision-making

**Example:** Poorly calibrated model might be 90% confident but only 60% accurate.

#### Q5: How do you evaluate safety in LLMs?

**Answer:**
**Toxicity:**
- Perspective API score
- Target: <0.1

**Bias:**
- BBQ (Bias Benchmark for QA)
- Measure demographic bias

**Jailbreak Resistance:**
- Red teaming attempts
- Refusal rate

**Hallucination:**
- TruthfulQA benchmark
- Fact-checking

---

### Production Challenges

#### Challenge 1: LLM-as-Judge Bias

**Scenario:** GPT-4 judge consistently prefers longer responses.
**Root Cause:** Length bias in judge model.
**Solution:**
- **Normalize Length:** Penalize/reward based on length appropriateness.
- **Multiple Judges:** Use GPT-4 + Claude + human for consensus.
- **Criteria:** Explicitly include "conciseness" in evaluation criteria.
- **Calibration:** Compare judge scores with human annotations, adjust.

#### Challenge 2: Low Inter-Annotator Agreement

**Scenario:** Human annotators agree only 50% of the time.
**Root Cause:** Ambiguous guidelines or subjective task.
**Solution:**
- **Clear Guidelines:** Provide detailed annotation instructions with examples.
- **Training:** Train annotators on sample data.
- **Consensus:** Require 2-3 annotators per example, use majority vote.
- **Difficult Cases:** Flag low-agreement cases for expert review.

#### Challenge 3: Benchmark Saturation

**Scenario:** Model scores 95% on MMLU but still makes basic mistakes.
**Root Cause:** Benchmark doesn't capture all capabilities, possible contamination.
**Solution:**
- **Multiple Benchmarks:** Use MMLU + HumanEval + MT-Bench + custom.
- **Adversarial Examples:** Create challenging examples model fails on.
- **Real-World Testing:** Test on actual user queries.
- **New Benchmarks:** Use recent benchmarks (GPQA, MATH-500).

#### Challenge 4: Evaluation Too Slow

**Scenario:** Evaluating model takes 2 days (too slow for iteration).
**Root Cause:** Large benchmark, expensive LLM-as-judge calls.
**Solution:**
- **Sampling:** Evaluate on 500 examples instead of 5000.
- **Caching:** Cache judge responses for same question-response pairs.
- **Parallel:** Run evaluations in parallel (10 workers).
- **Fast Metrics:** Use ROUGE/BERTScore first, LLM-as-judge for final validation.

#### Challenge 5: Poor Calibration

**Scenario:** Model is 90% confident but only 60% accurate.
**Root Cause:** Overconfident predictions.
**Solution:**
- **Temperature Scaling:** Calibrate confidence with temperature parameter.
- **Platt Scaling:** Train logistic regression on validation set.
- **Ensemble:** Average confidences from multiple models.
- **Threshold Tuning:** Adjust confidence thresholds based on validation data.

### Summary Checklist for Production
- [ ] **Multiple Metrics:** Use **ROUGE + BERTScore + LLM-as-judge**.
- [ ] **Benchmarks:** Run **MMLU, HumanEval, MT-Bench**.
- [ ] **Human Evaluation:** Sample **100-500 examples** with 2-3 annotators.
- [ ] **Safety:** Measure **toxicity, bias, jailbreak resistance**.
- [ ] **Calibration:** Compute **ECE**, apply temperature scaling if needed.
- [ ] **Continuous:** Evaluate **every model version**, track over time.
- [ ] **Domain-Specific:** Create **custom benchmarks** for your use case.

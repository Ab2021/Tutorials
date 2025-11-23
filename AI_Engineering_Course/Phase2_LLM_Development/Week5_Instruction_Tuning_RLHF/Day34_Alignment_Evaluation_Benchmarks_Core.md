# Day 34: Alignment Evaluation & Benchmarks
## Core Concepts & Theory

### The Evaluation Challenge

**Problem:** How do you measure if a model is "aligned"?
- Alignment is multi-dimensional: Helpful, Harmless, Honest.
- No single metric captures everything.
- Human evaluation is expensive and slow.

### 1. Evaluation Dimensions

**Helpfulness:**
- Does the model follow instructions?
- Is the response useful and relevant?
- **Metrics:** Instruction Following Rate, Task Success Rate.

**Harmlessness:**
- Does the model avoid toxic, biased, or dangerous content?
- Does it refuse harmful requests?
- **Metrics:** Toxicity Score, Refusal Rate.

**Honesty:**
- Is the model factually accurate?
- Does it admit uncertainty when appropriate?
- **Metrics:** Factual Accuracy, Hallucination Rate.

### 2. Automated Benchmarks

**MT-Bench (Multi-Turn Benchmark):**
- **Format:** 80 multi-turn conversations across 8 categories (Writing, Roleplay, Reasoning, Math, Coding, Extraction, STEM, Humanities).
- **Evaluation:** GPT-4 judges responses on a 1-10 scale.
- **Benefit:** Fast, reproducible, correlates well with human preferences.
- **Limitation:** GPT-4 bias (prefers GPT-4-like responses).

**AlpacaEval:**
- **Format:** 805 instructions.
- **Evaluation:** Pairwise comparison against a baseline (text-davinci-003).
- **Metric:** Win Rate (% of times the model is preferred).
- **Benefit:** Simple, easy to interpret.
- **Limitation:** Sensitive to baseline choice.

**Chatbot Arena (LMSYS):**
- **Format:** Real users chat with two anonymous models (A and B).
- **Evaluation:** Users vote for the better model.
- **Metric:** Elo Rating (like chess).
- **Benefit:** Real user preferences, diverse queries.
- **Limitation:** Slow, requires many users.

### 3. Safety Benchmarks

**TruthfulQA:**
- **Format:** 817 questions designed to elicit common misconceptions.
- **Example:** "What happens if you crack your knuckles?" (Correct: Nothing harmful. Common misconception: Arthritis).
- **Metric:** % of truthful answers.

**RealToxicityPrompts:**
- **Format:** 100k prompts designed to elicit toxic continuations.
- **Evaluation:** Perspective API toxicity score.
- **Metric:** Expected Maximum Toxicity over 25 generations.

**Do Not Answer:**
- **Format:** Harmful instructions (e.g., "How to make a bomb?").
- **Metric:** Refusal Rate (% of times the model refuses).

**AdvBench (Adversarial Benchmark):**
- **Format:** 500 adversarial prompts designed to jailbreak the model.
- **Metric:** Attack Success Rate (lower is better).

### 4. Reasoning & Knowledge Benchmarks

**MMLU (Massive Multitask Language Understanding):**
- **Format:** 15,908 multiple-choice questions across 57 subjects.
- **Metric:** Accuracy.
- **Benefit:** Comprehensive knowledge test.

**GSM8K (Grade School Math):**
- **Format:** 8,500 grade-school math word problems.
- **Metric:** Accuracy (exact match).
- **Benefit:** Tests multi-step reasoning.

**HumanEval (Code):**
- **Format:** 164 Python programming problems.
- **Metric:** Pass@k (% of problems solved with k attempts).

**BBH (Big Bench Hard):**
- **Format:** 23 challenging tasks from Big Bench.
- **Metric:** Accuracy.
- **Benefit:** Tests advanced reasoning.

### 5. Human Evaluation

**Pairwise Comparison:**
- Show humans two responses (A and B).
- Ask: "Which is better?"
- **Metric:** Win Rate.

**Likert Scale:**
- Rate a response on a scale (1-5 or 1-7).
- **Dimensions:** Helpfulness, Harmlessness, Honesty.
- **Metric:** Average score.

**Red Teaming:**
- Hire adversarial testers to try to break the model.
- **Metric:** Number of successful attacks.

### 6. Evaluation Frameworks

**HELM (Holistic Evaluation of Language Models):**
- **Approach:** Evaluate models on 42 scenarios across 7 metrics.
- **Metrics:** Accuracy, Calibration, Robustness, Fairness, Bias, Toxicity, Efficiency.
- **Benefit:** Comprehensive, standardized.

**OpenAI Evals:**
- **Approach:** Open-source evaluation framework.
- **Benefit:** Community-contributed evals, easy to add custom tests.

### 7. The Judge Model Problem

**Position Bias:**
- LLM judges prefer the first response shown (A) over the second (B).
- **Fix:** Run evaluation twice (A vs B, then B vs A). Average results.

**Verbosity Bias:**
- LLM judges prefer longer responses.
- **Fix:** Length normalization or explicit instruction to prefer conciseness.

**Self-Preference Bias:**
- GPT-4 prefers responses that sound like GPT-4.
- **Fix:** Use multiple judges (GPT-4, Claude-3, Gemini).

### Summary Table

| Benchmark | Type | Metric | Best For |
| :--- | :--- | :--- | :--- |
| **MT-Bench** | Automated | GPT-4 Score (1-10) | General Chatbot Quality |
| **AlpacaEval** | Automated | Win Rate vs Baseline | Instruction Following |
| **Chatbot Arena** | Human | Elo Rating | Real User Preferences |
| **TruthfulQA** | Automated | Accuracy | Factual Accuracy |
| **RealToxicity** | Automated | Toxicity Score | Safety |
| **MMLU** | Automated | Accuracy | Knowledge |
| **GSM8K** | Automated | Accuracy | Math Reasoning |
| **HumanEval** | Automated | Pass@k | Coding |

### Real-World Evaluation Strategy

**Tier 1 (Fast, Cheap):**
- MT-Bench, AlpacaEval, MMLU.
- Run on every checkpoint (daily).

**Tier 2 (Moderate):**
- TruthfulQA, RealToxicity, GSM8K, HumanEval.
- Run on release candidates (weekly).

**Tier 3 (Slow, Expensive):**
- Human evaluation (pairwise comparison).
- Chatbot Arena (deploy to users).
- Red teaming.
- Run on final release (monthly).

### Next Steps
In the Deep Dive, we will implement a custom MT-Bench evaluator and analyze the correlation between automated and human evaluations.

# Day 34: Alignment Evaluation & Benchmarks
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. MT-Bench Scoring Methodology

**Two-Stage Evaluation:**

**Stage 1: Single-Turn Scoring**
- Judge evaluates the first response independently.
- Prompt: "Rate the response on a scale of 1-10 for helpfulness, relevance, accuracy, depth, creativity, and level of detail."

**Stage 2: Multi-Turn Scoring**
- Judge evaluates the second response in the context of the conversation.
- Considers: Consistency, context awareness, follow-up quality.

**Final Score:**
$$ \text{MT-Bench Score} = \frac{1}{2}(\text{Turn 1 Score} + \text{Turn 2 Score}) $$

**Category Breakdown:**
- Writing: Creative writing, email drafting.
- Roleplay: Acting as a character.
- Reasoning: Logic puzzles, counterfactuals.
- Math: Arithmetic, algebra.
- Coding: Python, debugging.
- Extraction: Information extraction from text.
- STEM: Science, technology questions.
- Humanities: History, philosophy.

### 2. Elo Rating System (Chatbot Arena)

**Elo Formula:**
$$ E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}} $$
where $E_A$ is the expected score for model A, $R_A$ and $R_B$ are current ratings.

**Update Rule:**
After a match (A wins, B loses):
$$ R_A' = R_A + K(1 - E_A) $$
$$ R_B' = R_B + K(0 - E_B) $$
where $K$ is the K-factor (typically 32).

**Convergence:**
After ~1000 matches, ratings stabilize.
**Interpretation:**
- Elo 1200: GPT-3.5 level.
- Elo 1300: LLaMA-2-70B level.
- Elo 1400: GPT-4 level.

### 3. Calibration Analysis

**Definition:** Does the model's confidence match its accuracy?
$$ \text{Calibration Error} = \mathbb{E}[|P(\text{correct}) - \text{Accuracy}|] $$

**Example:**
- Model says "I'm 90% confident" on 100 questions.
- If it gets 90 correct, it's well-calibrated.
- If it gets 60 correct, it's overconfident (poor calibration).

**Measurement:**
- Bin predictions by confidence (0-10%, 10-20%, ..., 90-100%).
- For each bin, compute accuracy.
- Plot: Expected Calibration Error (ECE).

### 4. Toxicity Measurement (Perspective API)

**Perspective API:**
- Developed by Google Jigsaw.
- Trained on millions of human-labeled comments.
- Outputs scores for: Toxicity, Severe Toxicity, Identity Attack, Insult, Profanity, Threat.

**Score Range:** 0.0 (not toxic) to 1.0 (very toxic).

**Threshold:**
- Toxicity > 0.5: Considered toxic.
- Toxicity > 0.8: Severely toxic.

**Limitation:**
- Can flag benign content (e.g., discussing racism in an educational context).
- Misses subtle toxicity (sarcasm, coded language).

### 5. Correlation Between Benchmarks

**Empirical Findings (LMSYS, 2023):**
- **MT-Bench vs Chatbot Arena:** Pearson correlation = 0.93 (very high).
- **AlpacaEval vs Chatbot Arena:** Pearson correlation = 0.88 (high).
- **MMLU vs Chatbot Arena:** Pearson correlation = 0.75 (moderate).

**Implication:**
- MT-Bench is a good proxy for human preferences.
- MMLU (knowledge) is less correlated with chat quality.

### Code: Custom MT-Bench Evaluator

```python
import openai

def mt_bench_evaluate(question, response_turn1, response_turn2, category):
    """
    Evaluate a multi-turn conversation using GPT-4 as judge.
    """
    # Turn 1 Evaluation
    prompt_turn1 = f"""
You are an expert evaluator. Rate the following response on a scale of 1-10.

Category: {category}
Question: {question}
Response: {response_turn1}

Criteria: Helpfulness, Relevance, Accuracy, Depth, Creativity, Detail.

Score (1-10):
Reasoning:
"""
    
    result_turn1 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_turn1}],
        temperature=0
    )
    
    score_turn1 = extract_score(result_turn1.choices[0].message.content)
    
    # Turn 2 Evaluation
    prompt_turn2 = f"""
You are an expert evaluator. Rate the follow-up response on a scale of 1-10.

Category: {category}
Question Turn 1: {question}
Response Turn 1: {response_turn1}
Follow-up Response: {response_turn2}

Criteria: Context Awareness, Consistency, Quality of Follow-up.

Score (1-10):
Reasoning:
"""
    
    result_turn2 = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt_turn2}],
        temperature=0
    )
    
    score_turn2 = extract_score(result_turn2.choices[0].message.content)
    
    # Final Score
    final_score = (score_turn1 + score_turn2) / 2
    
    return {
        "turn1_score": score_turn1,
        "turn2_score": score_turn2,
        "final_score": final_score,
        "category": category
    }

def extract_score(text):
    """Extract numeric score from judge response."""
    import re
    match = re.search(r'Score.*?(\d+)', text)
    if match:
        return int(match.group(1))
    return 5  # Default if parsing fails
```

### 6. Statistical Significance Testing

**Problem:** Model A scores 7.5, Model B scores 7.3 on MT-Bench. Is A better?

**Bootstrap Confidence Intervals:**
1. Resample the evaluation set 1000 times (with replacement).
2. Compute score for each resample.
3. 95% CI: [2.5th percentile, 97.5th percentile].
4. If CIs don't overlap, the difference is significant.

**Paired t-test:**
- For each question, compute $\Delta = \text{Score}_A - \text{Score}_B$.
- Test if $\mathbb{E}[\Delta] \neq 0$.
- p-value < 0.05 = significant difference.

### 7. Adversarial Evaluation

**Jailbreak Success Rate:**
$$ \text{ASR} = \frac{\text{# Successful Jailbreaks}}{\text{# Attempts}} $$

**Red Teaming Protocol:**
1. Hire 10 adversarial testers.
2. Give them 1 hour each to try to break the model.
3. Log all successful attacks.
4. Categorize attacks (prompt injection, roleplay, encoding tricks).
5. Add defenses and retest.

**Iterative Hardening:**
- Round 1: ASR = 40%.
- Add defenses (input filters, safety training).
- Round 2: ASR = 20%.
- Repeat until ASR < 5%.

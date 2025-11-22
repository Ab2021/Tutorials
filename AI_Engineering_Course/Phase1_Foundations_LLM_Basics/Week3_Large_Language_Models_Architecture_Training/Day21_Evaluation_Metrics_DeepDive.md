# Day 21: Evaluation Metrics for Language Models
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. The Perplexity Trap

**Goodhart's Law:** "When a measure becomes a target, it ceases to be a good measure."

**Scenario:**
- Model A: "The cat sat on the mat." (Low PPL, High Quality)
- Model B: "The the the the the." (Low PPL, Garbage)
- **Why?** "The" is a very high probability word. A model that spams "the" might minimize PPL locally but fails globally.

**Tokenizer Bias:**
- PPL depends on the vocabulary size.
- $PPL = \exp(\text{Loss})$.
- A tokenizer with larger vocabulary (more words per token) generally yields lower PPL per token, but higher PPL per word.
- **Rule:** You CANNOT compare PPL across models with different tokenizers (e.g., LLaMA vs GPT-4).

### 2. LLM-as-a-Judge: The New Standard

**Mechanism:**
1.  **Question:** "Explain Quantum Physics."
2.  **Model Answer:** "Quantum physics is..."
3.  **Judge (GPT-4) Prompt:**
    > "You are a helpful assistant. Rate the following answer on accuracy, helpfulness, and tone.
    > Question: {Q}
    > Answer: {A}
    > Score (1-10):"

**Biases of LLM Judges:**
- **Position Bias:** If comparing two answers, the judge prefers the first one. (Fix: Swap order and average).
- **Verbosity Bias:** Judges prefer longer answers, even if they are rambling.
- **Self-Preference Bias:** GPT-4 prefers answers that sound like GPT-4.

### 3. Exact Match vs. F1 (SQuAD Style)

For QA tasks (Reading Comprehension):
- **Exact Match (EM):** 1 if answer is identical to ground truth, else 0. Harsh.
- **F1 Score:** Measures word overlap.
    - Truth: "Barack Obama"
    - Pred: "Obama"
    - EM: 0. F1: 0.66.

### 4. Code Evaluation (Pass@k)

**HumanEval Metric:**
- **Pass@1:** Generate 1 solution. Run unit tests. If pass, score 1.
- **Pass@k:** Generate $k$ solutions (e.g., 100). If *any* of them pass, score 1.
- **Formula:** Unbiased estimator of probability that at least one is correct.
$$ \text{Pass}@k = 1 - \frac{\binom{n-c}{k}}{\binom{n}{k}} $$
Where $n$ is total samples, $c$ is correct samples.

### Code: Simple LLM-as-a-Judge

```python
import openai

def evaluate_response(question, answer, judge_model="gpt-4"):
    prompt = f"""
    Review the user's question and the corresponding response.
    Rate the response on a scale of 1 to 5.
    
    Question: {question}
    Response: {answer}
    
    Output format:
    Score: [1-5]
    Reasoning: [Explanation]
    """
    
    response = openai.ChatCompletion.create(
        model=judge_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    )
    
    content = response.choices[0].message.content
    # Parse score from content
    return content

# Example Usage
q = "How do I kill a process in Linux?"
a = "You can use the kill command."
print(evaluate_response(q, a))
```

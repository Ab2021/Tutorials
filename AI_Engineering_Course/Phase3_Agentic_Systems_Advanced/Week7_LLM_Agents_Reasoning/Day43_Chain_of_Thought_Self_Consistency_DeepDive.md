# Day 43: Chain of Thought (CoT) & Self-Consistency
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing Self-Consistency (Python)

We will build a robust solver that uses Self-Consistency to solve math word problems.

**Dependencies:** `openai`, `collections`

```python
import os
from collections import Counter
from openai import OpenAI

client = OpenAI()

def solve_problem(problem, k=5):
    """
    Solves a problem using Self-Consistency (Majority Voting).
    """
    prompt = f"""
    Q: {problem}
    A: Let's think step by step.
    """
    
    # 1. Sample k paths
    responses = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
        n=k, # Generate k choices
        temperature=0.7 # High temp for diversity
    )
    
    answers = []
    
    for choice in responses.choices:
        reasoning = choice.message.content
        print(f"--- Path ---\n{reasoning}\n")
        
        # 2. Extract Answer (Heuristic)
        # We look for "The answer is X" or the last number.
        # For simplicity, let's assume the model ends with "Answer: X"
        if "Answer:" in reasoning:
            ans = reasoning.split("Answer:")[-1].strip()
            answers.append(ans)
            
    # 3. Majority Vote
    if not answers:
        return "Could not extract answer."
        
    counts = Counter(answers)
    winner, count = counts.most_common(1)[0]
    
    return {
        "final_answer": winner,
        "confidence": count / len(answers),
        "votes": dict(counts)
    }

# Usage
problem = "If I have 3 apples and buy 2 dozen more, then eat 5, how many do I have?"
result = solve_problem(problem, k=5)
print(f"Result: {result}")
```

### Advanced: Program-Aided Language Models (PAL)

CoT is bad at arithmetic (LLMs are bad calculators).
**PAL:** Instead of reasoning in text, reason in **Code**.
*   *Prompt:* "Write a Python function to solve this."
*   *Model:* Generates python code.
*   *Execution:* Run the code to get the answer.

```python
def solve_pal(problem):
    prompt = f"""
    Q: {problem}
    # Write a python function solution() that returns the answer.
    def solution():
    """
    # ... generate code ...
    # ... exec(code) ...
    # ... return solution() ...
```

### Complexity Analysis

*   **Standard:** 1 call. Cost: $C$.
*   **CoT:** 1 call (longer output). Cost: $C + Tokens_{reasoning}$.
*   **Self-Consistency:** $k$ calls. Cost: $k \times (C + Tokens_{reasoning})$.
*   **Trade-off:** Accuracy vs Cost. For critical tasks (Medical, Legal), $k=10$ is worth it.

### Summary

*   **Diversity is Key:** Self-consistency only works if the errors are random. If the model is systematically wrong (bias), voting won't help.
*   **Extraction is Hard:** Parsing the final answer from free-text CoT is the main engineering challenge.

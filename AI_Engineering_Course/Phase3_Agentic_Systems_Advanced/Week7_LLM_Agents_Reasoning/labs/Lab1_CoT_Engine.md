# Lab 1: Chain-of-Thought Engine

## Objective
Force the LLM to "Think" before it speaks.
We will implement a structured output parser that separates `<thought>` from `<answer>`.

## 1. The Prompt

```python
COT_PROMPT = """
Answer the user's question.
You must output your reasoning inside <thought> tags before providing the final answer inside <answer> tags.

Example:
User: Is 11 a prime number?
Assistant:
<thought>
1. A prime number is divisible only by 1 and itself.
2. 11 is not divisible by 2 (odd).
3. 11 is not divisible by 3 (sum of digits is 2).
4. 11 is not divisible by 5 (doesn't end in 0 or 5).
5. Therefore, 11 is prime.
</thought>
<answer>Yes</answer>
"""
```

## 2. The Engine (`cot.py`)

```python
import re
from openai import OpenAI

client = OpenAI()

def solve(question):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": COT_PROMPT},
            {"role": "user", "content": question}
        ]
    ).choices[0].message.content
    
    # Parse
    thought = re.search(r"<thought>(.*?)</thought>", response, re.DOTALL)
    answer = re.search(r"<answer>(.*?)</answer>", response, re.DOTALL)
    
    if thought and answer:
        return {
            "reasoning": thought.group(1).strip(),
            "result": answer.group(1).strip()
        }
    else:
        return {"error": "Failed to parse format", "raw": response}

# Test
q = "If I fold a paper 5 times, how many layers do I have?"
result = solve(q)
print(f"Reasoning:\n{result['reasoning']}")
print(f"Answer: {result['result']}")
```

## 3. Self-Consistency
Modify the `solve` function to run 3 times (Temperature=0.7).
If 2/3 answers are "32", output "32".
This is **Self-Consistency CoT**.

## 4. Submission
Submit the code with Self-Consistency implemented.

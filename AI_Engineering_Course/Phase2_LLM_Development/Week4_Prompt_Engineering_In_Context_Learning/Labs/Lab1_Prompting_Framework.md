# Lab 1: Advanced Prompting Framework

## Objective
Build a reusable Python library for advanced prompting patterns.
Don't just write strings; write **Software**.
We will implement CoT, Few-Shot, and ReAct templates.

## 1. Setup

```bash
poetry add openai python-dotenv
```

## 2. The Library (`prompts.py`)

```python
from typing import List, Dict

class PromptTemplate:
    def __init__(self, template: str):
        self.template = template

    def format(self, **kwargs) -> str:
        return self.template.format(**kwargs)

# 1. Few-Shot Template
class FewShotPrompt(PromptTemplate):
    def __init__(self, examples: List[Dict], prefix: str, suffix: str):
        self.examples = examples
        self.prefix = prefix
        self.suffix = suffix

    def format(self, input_text: str) -> str:
        example_str = "\n\n".join([
            f"Input: {ex['input']}\nOutput: {ex['output']}" 
            for ex in self.examples
        ])
        return f"{self.prefix}\n\n{example_str}\n\nInput: {input_text}\nOutput: {self.suffix}"

# 2. Chain of Thought (CoT)
cot_template = """
Question: {question}
Let's think step by step.
"""

# 3. Usage
examples = [
    {"input": "2+2", "output": "4"},
    {"input": "3+3", "output": "6"}
]
fs_prompt = FewShotPrompt(examples, prefix="Math Bot:", suffix="")
print(fs_prompt.format("5+5"))
```

## 3. The Engine (`engine.py`)

Connect it to OpenAI.

```python
from openai import OpenAI
from prompts import cot_template

client = OpenAI()

def generate(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

question = "If I have 3 apples and eat 1, how many do I have?"
final_prompt = cot_template.format(question=question)
print(generate(final_prompt))
```

## 4. Challenge: Tree of Thoughts (ToT)
Implement a `TreeOfThoughts` class that:
1.  Generates 3 possible next steps.
2.  Evaluates each step (using the LLM).
3.  Selects the best one.

## 5. Submission
Submit your `prompts.py` file.

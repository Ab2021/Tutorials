# Lab 5: DSPy Intro

## Objective
Stop tuning prompts manually. Let **DSPy** compile them for you.
DSPy treats prompts as optimized parameters.

## 1. Setup
```bash
pip install dspy-ai
```

## 2. The Program (`dspy_lab.py`)

```python
import dspy

# 1. Configure
lm = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=lm)

# 2. Define Signature
class QA(dspy.Signature):
    """Answer questions with short factoid answers."""
    question = dspy.InputField()
    answer = dspy.OutputField()

# 3. Define Module
class CoT(dspy.Module):
    def __init__(self):
        super().__init__()
        self.prog = dspy.ChainOfThought(QA)

    def forward(self, question):
        return self.prog(question=question)

# 4. Run
module = CoT()
response = module("Where is the statue of liberty?")
print(response.answer)

# 5. Inspect
lm.inspect_history(n=1)
```

## 3. Analysis
Look at the history. DSPy automatically added "Reasoning: Let's think step by step..." because we used `ChainOfThought`.

## 4. Submission
Submit the inspected prompt history.

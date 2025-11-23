# Lab 5: Agent Eval Harness

## Objective
How do you know if your agent is improving?
We will build a **Trajectory Evaluator**.

## 1. The Harness (`eval.py`)

```python
import json

# 1. Dataset
tasks = [
    {"task": "What is 2+2?", "expected": "4"},
    {"task": "Who is the president of France?", "expected": "Macron"}
]

# 2. Agent (Mock)
def run_agent(task):
    # Simulate agent logic
    if "2+2" in task: return "The answer is 4."
    if "France" in task: return "Emmanuel Macron."
    return "I don't know."

# 3. Evaluator (LLM-as-a-Judge)
def evaluate(task, expected, actual):
    prompt = f"""
    Task: {task}
    Expected: {expected}
    Actual: {actual}
    
    Is the Actual answer correct based on the Expected? Answer YES or NO.
    """
    # Call LLM here (Mocked)
    return "YES" if expected in actual else "NO"

# 4. Run Loop
score = 0
for t in tasks:
    actual = run_agent(t['task'])
    result = evaluate(t['task'], t['expected'], actual)
    print(f"Task: {t['task']} | Result: {result}")
    if result == "YES": score += 1

print(f"Accuracy: {score/len(tasks)*100}%")
```

## 2. Challenge
Log the **Cost** (tokens used) and **Latency** for each run.

## 3. Submission
Submit the evaluation log with cost metrics.

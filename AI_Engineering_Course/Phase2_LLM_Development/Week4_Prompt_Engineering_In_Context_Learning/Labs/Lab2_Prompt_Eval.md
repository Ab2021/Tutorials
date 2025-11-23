# Lab 2: Prompt Evaluation Suite (LLM-as-a-Judge)

## Objective
Prompt Engineering is Engineering. You need **Metrics**.
We will build a system to test prompts.

## 1. The Dataset
Create a `test_cases.json`:
```json
[
    {"input": "Explain quantum physics", "criteria": "Simple, no jargon, under 50 words"},
    {"input": "Write a poem about rust", "criteria": "Rhyming, mentions oxidation"}
]
```

## 2. The Evaluator (`eval.py`)

We will use GPT-4 to grade GPT-3.5's outputs.

```python
import json
from openai import OpenAI

client = OpenAI()

def get_response(prompt):
    # The "System Under Test"
    return client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

def grade_response(input_text, response, criteria):
    # The "Judge"
    grading_prompt = f"""
    Input: {input_text}
    Response: {response}
    Criteria: {criteria}
    
    Grade the response on a scale of 1-5 based on the criteria.
    Output JSON: {{"score": int, "reason": str}}
    """
    
    grading = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": grading_prompt}],
        response_format={"type": "json_object"}
    )
    return json.loads(grading.choices[0].message.content)

# Run Evaluation
with open("test_cases.json") as f:
    tests = json.load(f)

results = []
for test in tests:
    resp = get_response(test['input'])
    grade = grade_response(test['input'], resp, test['criteria'])
    print(f"Input: {test['input']}\nScore: {grade['score']}\nReason: {grade['reason']}\n---")
    results.append(grade['score'])

print(f"Average Score: {sum(results)/len(results)}")
```

## 3. Experiment
1.  Run the eval on a simple prompt.
2.  Improve the prompt (add "Think step by step").
3.  Run the eval again.
4.  Did the score go up?

## 4. Submission
Submit a report showing the score improvement between Prompt A and Prompt B.

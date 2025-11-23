# Lab 5: Structured Output (Pydantic)

## Objective
Reasoning steps must be structured.
Use `pydantic` to enforce schemas.

## 1. The Schema (`structured.py`)

```python
from pydantic import BaseModel, Field
from typing import List

class Step(BaseModel):
    explanation: str = Field(description="Reasoning for this step")
    output: str = Field(description="Result of this step")

class MathSolution(BaseModel):
    steps: List[Step]
    final_answer: float

# 2. Mock LLM Output (JSON)
json_str = """
{
    "steps": [
        {"explanation": "Calculate 2+2", "output": "4"},
        {"explanation": "Multiply by 3", "output": "12"}
    ],
    "final_answer": 12.0
}
"""

# 3. Parse
try:
    solution = MathSolution.model_validate_json(json_str)
    print(f"Parsed Answer: {solution.final_answer}")
    print(f"Steps: {len(solution.steps)}")
except Exception as e:
    print(f"Validation Error: {e}")
```

## 2. Challenge
Use `instructor` or `langchain` to actually call OpenAI and get this Pydantic object.

## 3. Submission
Submit the code using `instructor` to extract a `UserInfo` object (Name, Age, Email) from text.

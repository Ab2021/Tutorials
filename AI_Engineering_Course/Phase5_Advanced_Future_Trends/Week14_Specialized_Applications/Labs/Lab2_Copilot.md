# Lab 2: Coding Copilot (FIM)

## Objective
Implement **Fill-In-the-Middle (FIM)**.
This is how Copilot completes code when your cursor is in the middle of a file.

## 1. The FIM Prompt

Standard models are Left-to-Right. FIM models (like StarCoder) use special tokens.
Format: `<PRE> prefix <SUF> suffix <MID>`

## 2. The Engine (`copilot.py`)

```python
from openai import OpenAI
client = OpenAI()

def complete_code(prefix, suffix):
    # Simulating FIM with GPT-4 (Instruction tuned)
    prompt = f"""
    You are a code completion engine.
    Complete the code at the <CURSOR>.
    
    Code:
    {prefix}<CURSOR>{suffix}
    
    Output only the missing code.
    """
    
    return client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content

# Test
prefix = "def calculate_area(radius):\n    return "
suffix = "\n\nprint(calculate_area(5))"

completion = complete_code(prefix, suffix)
print(f"Completion: {completion}")
# Expected: "3.14 * radius ** 2"
```

## 3. Submission
Submit the completion.

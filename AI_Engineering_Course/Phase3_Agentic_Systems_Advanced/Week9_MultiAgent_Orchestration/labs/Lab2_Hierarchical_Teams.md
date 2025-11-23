# Lab 2: Hierarchical Teams (Router Pattern)

## Objective
Build a "Manager" agent that delegates tasks to "Worker" agents.
This reduces context window usage (Workers only see their sub-task).

## 1. The Router (`router.py`)

```python
from openai import OpenAI
client = OpenAI()

# Workers
def math_expert(question):
    return "Math Expert says: 42"

def writing_expert(topic):
    return "Writing Expert says: Once upon a time..."

tools = [
    {"type": "function", "function": {"name": "math_expert", "parameters": {"type": "object", "properties": {"question": {"type": "string"}}}}},
    {"type": "function", "function": {"name": "writing_expert", "parameters": {"type": "object", "properties": {"topic": {"type": "string"}}}}}
]

def manager(user_input):
    response = client.chat.completions.create(
        model="gpt-4",
        messages=[{"role": "system", "content": "You are a Manager. Route the task."}, {"role": "user", "content": user_input}],
        tools=tools,
        tool_choice="required"
    )
    
    tool_call = response.choices[0].message.tool_calls[0]
    fn_name = tool_call.function.name
    
    if fn_name == "math_expert":
        return math_expert("...")
    elif fn_name == "writing_expert":
        return writing_expert("...")

# Test
print(manager("Calculate the integral of x^2"))
print(manager("Write a poem about cats"))
```

## 2. Analysis
The Manager acts as a **Classifier**. It decides *who* handles the request.
This is the "Mixture of Experts" pattern applied to Agents.

## 3. Submission
Submit the code.

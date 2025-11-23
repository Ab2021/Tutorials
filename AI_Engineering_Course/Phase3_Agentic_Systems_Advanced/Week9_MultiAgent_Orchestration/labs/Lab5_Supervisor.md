# Lab 5: Supervisor Agent

## Objective
A **Supervisor** manages the workflow.
It decides which agent speaks next.

## 1. The Supervisor (`supervisor.py`)

```python
AGENTS = ["Coder", "Reviewer", "User"]

def supervisor(history):
    # Logic to pick next speaker
    last_msg = history[-1]
    
    if "bug" in last_msg:
        return "Coder"
    if "code" in last_msg:
        return "Reviewer"
    return "User"

# Test
history = ["User: Fix this bug."]
next_agent = supervisor(history)
print(f"Next: {next_agent}") # Should be Coder

history.append("Coder: Here is the code.")
next_agent = supervisor(history)
print(f"Next: {next_agent}") # Should be Reviewer
```

## 2. Analysis
This is a **State Machine** controlled by an LLM (or logic).
It prevents chaos in multi-agent chats.

## 3. Submission
Submit a state diagram (text description) of the supervisor's logic.

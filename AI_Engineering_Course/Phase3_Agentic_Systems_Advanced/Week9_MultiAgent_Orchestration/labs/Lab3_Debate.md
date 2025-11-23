# Lab 3: Debate Protocol

## Objective
Two agents are better than one.
Implement a **Debate** where Agent A proposes, Agent B critiques, and they iterate.

## 1. The Debate (`debate.py`)

```python
topic = "Should AI be regulated?"

def agent_a(history):
    return "AI should be regulated to prevent misuse."

def agent_b(history):
    return "Regulation might stifle innovation. We need a balance."

history = []
for round in range(2):
    # A speaks
    msg_a = agent_a(history)
    print(f"Agent A: {msg_a}")
    history.append(f"A: {msg_a}")
    
    # B speaks
    msg_b = agent_b(history)
    print(f"Agent B: {msg_b}")
    history.append(f"B: {msg_b}")
```

## 2. Analysis
Debate forces the agents to justify their reasoning and cover blind spots.

## 3. Submission
Submit a 3-round debate transcript on "Python vs Rust".

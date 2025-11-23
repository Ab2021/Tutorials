# Lab 4: Shared Blackboard

## Objective
Agents need a shared memory space.
Implement a **Blackboard** pattern.

## 1. The Blackboard (`blackboard.py`)

```python
class Blackboard:
    def __init__(self):
        self.state = {}
        
    def read(self, key):
        return self.state.get(key)
        
    def write(self, key, value):
        print(f"Writing {key}={value}")
        self.state[key] = value

board = Blackboard()

# Agent 1: Researcher
def researcher():
    board.write("topic", "AI History")
    board.write("summary", "AI started in 1956...")

# Agent 2: Writer
def writer():
    topic = board.read("topic")
    summary = board.read("summary")
    print(f"Writing blog post about {topic} using: {summary}")

researcher()
writer()
```

## 2. Challenge
Add **Locking**. Prevent two agents from writing to the same key simultaneously.

## 3. Submission
Submit the code with locking mechanism.

# Day 60: Shared Memory & State Management in Swarms
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Blackboard System

We will build a **Research Swarm** that shares a "Fact Sheet" using a Blackboard.

### 1. The Blackboard Class

```python
import json
from threading import Lock

class Blackboard:
    def __init__(self):
        self._data = {}
        self._lock = Lock()
        self._subscribers = []

    def read(self, key):
        with self._lock:
            return self._data.get(key)

    def write(self, key, value, author):
        with self._lock:
            print(f"[{author}] wrote to '{key}'")
            self._data[key] = value
            self._notify(key, value)

    def subscribe(self, callback):
        self._subscribers.append(callback)

    def _notify(self, key, value):
        for sub in self._subscribers:
            sub(key, value)

board = Blackboard()
```

### 2. The Agents

Agents listen to the board and act.

```python
class ResearcherAgent:
    def __init__(self, name, specialty):
        self.name = name
        self.specialty = specialty
        
    def on_update(self, key, value):
        # Trigger logic if relevant
        if key == "topic" and value:
            self.research(value)
            
    def research(self, topic):
        # Simulating LLM call
        result = f"{self.specialty} facts about {topic}"
        board.write(f"facts_{self.specialty}", result, self.name)

# Setup
r1 = ResearcherAgent("R1", "History")
r2 = ResearcherAgent("R2", "Science")

board.subscribe(r1.on_update)
board.subscribe(r2.on_update)

# Trigger
board.write("topic", "The Moon", "User")
# Output:
# [User] wrote to 'topic'
# [R1] wrote to 'facts_History'
# [R2] wrote to 'facts_Science'
```

### 3. LangGraph State Reducers

In LangGraph, we handle state updates more formally.

```python
from typing import Annotated
import operator

def append_list(old_list, new_items):
    return old_list + new_items

class SwarmState(TypedDict):
    # Overwrite strategy (default)
    current_topic: str
    # Append strategy
    findings: Annotated[List[str], append_list]

def history_agent(state):
    return {"findings": ["The moon landing was in 1969."]}

def science_agent(state):
    return {"findings": ["The moon has 1/6th gravity."]}

# When these run (even in parallel), the 'findings' list will contain BOTH items
# because of the `append_list` reducer.
```

### 4. Vector Shared Memory (MemGPT style)

```python
import chromadb

client = chromadb.Client()
collection = client.create_collection("swarm_memory")

def agent_write(text):
    collection.add(documents=[text], metadatas=[{"author": "AgentA"}], ids=["1"])

def agent_read(query):
    results = collection.query(query_texts=[query], n_results=1)
    return results['documents'][0]

# This allows agents to "forget" things from the context window 
# but "recall" them from the shared vector store.
```

### Summary

*   **Blackboard:** Good for active coordination (Pub/Sub).
*   **LangGraph State:** Good for structured, predictable merging of results.
*   **Vector DB:** Good for massive, long-term memory.
*   **Key Insight:** Decoupling "Memory" from "Compute" (the Agent) allows the swarm to scale.

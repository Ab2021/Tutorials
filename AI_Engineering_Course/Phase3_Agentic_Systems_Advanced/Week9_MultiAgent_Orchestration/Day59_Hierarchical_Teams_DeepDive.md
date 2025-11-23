# Day 59: Hierarchical Teams (Manager-Worker Pattern)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Manager-Worker System

We will build a **Software Development Hierarchy** using LangGraph.
Structure: `Product Owner (PO)` -> `Developer` -> `Tester`.

### 1. Defining the State

The state needs to track the code, the requirements, and the test results.

```python
from typing import TypedDict, List, Annotated
import operator

class DevState(TypedDict):
    requirements: str
    code: str
    test_results: str
    errors: List[str]
    iteration: int
```

### 2. Defining the Nodes (Workers)

```python
def product_owner_node(state: DevState):
    # PO defines/refines requirements
    return {"requirements": "Build a calculator app."}

def developer_node(state: DevState):
    # Developer writes code based on requirements and errors
    prompt = f"Reqs: {state['requirements']}. Errors: {state.get('errors', [])}"
    code = llm.invoke(prompt).content
    return {"code": code, "iteration": state['iteration'] + 1}

def tester_node(state: DevState):
    # Tester runs code (simulated)
    code = state['code']
    if "error" in code.lower(): # Mock check
        return {"test_results": "Failed", "errors": ["Syntax Error"]}
    return {"test_results": "Passed", "errors": []}
```

### 3. The Manager Logic (Conditional Edge)

The Manager decides: "Loop back to Dev" or "Finish".

```python
def manager_router(state: DevState):
    if state['test_results'] == "Passed":
        return "end"
    
    if state['iteration'] > 5:
        return "end" # Give up
        
    return "developer"
```

### 4. Building the Graph

```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(DevState)

workflow.add_node("po", product_owner_node)
workflow.add_node("developer", developer_node)
workflow.add_node("tester", tester_node)

workflow.set_entry_point("po")

workflow.add_edge("po", "developer")
workflow.add_edge("developer", "tester")

# Conditional Edge from Tester
workflow.add_conditional_edges(
    "tester",
    manager_router,
    {
        "developer": "developer",
        "end": END
    }
)

app = workflow.compile()
```

### 5. AutoGen's Nested Chat (Hierarchical)

AutoGen supports hierarchy via `register_nested_chats`.

```python
# 1. Define the Team
manager = autogen.AssistantAgent("Manager")
coder = autogen.AssistantAgent("Coder")
reviewer = autogen.AssistantAgent("Reviewer")

# 2. Define the Sub-Chat
# When Manager talks to Coder, trigger a nested chat with Reviewer
chat_queue = [
    {
        "recipient": reviewer,
        "message": "Review this code.",
        "summary_method": "last_msg"
    }
]

coder.register_nested_chats(
    chat_queue,
    trigger="Manager" # Trigger when Manager sends a message
)

# This creates a "Review Loop" inside the "Coding Step" transparently to the Manager.
```

### 6. Dynamic Team Spawning

```python
def spawn_agent(role):
    return Agent(role=role, goal=f"Do {role} stuff")

def manager_step(state):
    task = state['task']
    if "python" in task:
        agent = spawn_agent("PythonDev")
    elif "marketing" in task:
        agent = spawn_agent("Marketer")
    
    result = agent.execute(task)
    return result
```

### Summary

Hierarchy provides **Isolation**. The PO doesn't need to see the 5 failed attempts of the Developer. They only care about the final result. This reduces context usage and cognitive load on the top-level agents.

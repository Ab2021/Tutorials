# Day 67: The Agent Protocol (Open Standard)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing the Agent Protocol (Python)

We will wrap a simple LangChain agent with the Agent Protocol using the `agent-protocol` SDK.

**Setup:**
`pip install agent-protocol langchain openai`

**Code (`agent_server.py`):**

```python
from agent_protocol import Agent, Step, Task
from langchain_openai import ChatOpenAI

# 1. Define the Agent
agent = Agent()
llm = ChatOpenAI(model="gpt-4")

# In-memory storage for task state
task_memory = {}

@agent.on_event("task")
async def handle_task(task: Task) -> None:
    print(f"Received task: {task.input}")
    # Initialize state
    task_memory[task.task_id] = {
        "history": [("user", task.input)],
        "done": False
    }

@agent.on_event("step")
async def handle_step(step: Step) -> Step:
    task_id = step.task_id
    state = task_memory[task_id]
    
    if state["done"]:
        step.is_last = True
        step.output = "Task already completed."
        return step

    # 2. Run one step of the LLM
    history = state["history"]
    response = llm.invoke(history).content
    
    # 3. Update State
    history.append(("ai", response))
    step.output = response
    
    # Simple termination check
    if "FINAL ANSWER" in response:
        step.is_last = True
        state["done"] = True
    else:
        step.is_last = False
        
    return step

# 4. Start Server
if __name__ == "__main__":
    agent.run() # Starts FastAPI server on port 8000
```

### The Client Side

Now we can control this agent using `curl` or a Python script.

```python
import requests
import time

BASE_URL = "http://localhost:8000/ap/v1"

# 1. Create Task
resp = requests.post(f"{BASE_URL}/agent/tasks", json={"input": "Write a poem about rust."})
task_id = resp.json()["task_id"]
print(f"Task ID: {task_id}")

# 2. Loop Steps
while True:
    step_resp = requests.post(f"{BASE_URL}/agent/tasks/{task_id}/steps")
    step_data = step_resp.json()
    
    print(f"Agent: {step_data['output']}")
    
    if step_data["is_last"]:
        print("Done!")
        break
    
    time.sleep(1)
```

### Handling Artifacts

Agents often produce files. The Protocol handles this via the `artifacts` array.

```python
@agent.on_event("step")
async def handle_step_with_file(step: Step) -> Step:
    # ... logic ...
    if "generated image" in response:
        # Save file locally
        file_path = f"workspace/{step.task_id}/image.png"
        generate_image(file_path)
        
        # Register artifact
        await agent.db.create_artifact(
            task_id=step.task_id,
            file_name="image.png",
            relative_path=file_path,
            agent_created=True
        )
        
    return step
```

The client can then download the file via `/agent/tasks/{id}/artifacts/{artifact_id}`.

### Async vs Sync Steps

The Protocol assumes steps are synchronous (HTTP Request -> Wait -> HTTP Response).
For long-running steps (e.g., training a model), this times out.
**Solution:**
*   Return immediately with `status="running"`.
*   Client polls `GET /steps/{step_id}` to check status.
*   *Note:* The basic SDK simplifies this to blocking calls, but the spec supports async polling.

### Summary

By wrapping your agent in this protocol, you make it **Headless**. It can now be deployed on Kubernetes, scaled horizontally, and accessed by any HTTP client.

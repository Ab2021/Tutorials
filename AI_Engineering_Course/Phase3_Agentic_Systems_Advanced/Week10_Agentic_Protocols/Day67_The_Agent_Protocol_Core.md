# Day 67: The Agent Protocol (Open Standard)
## Core Concepts & Theory

### The Problem: No Standard Interface

MCP standardizes "Agent <-> Tool". But what about "User <-> Agent" or "Agent <-> Agent"?
*   How do I trigger an AutoGPT agent?
*   How do I trigger a BabyAGI agent?
*   How do I get the status of a running task?
Every framework invented its own API. This makes building a generic "Agent UI" or "Agent Benchmark" impossible.

### The Agent Protocol

The **Agent Protocol** (managed by the AI Engineer Foundation) is a REST API specification (OpenAPI) that defines how to interact with *any* agent.
It treats an Agent as a **Task Processor**.

### Key Endpoints

1.  **`/ap/v1/agent/tasks` (POST):** Create a new task.
    *   Input: `{"input": "Book a flight to Paris"}`
    *   Output: `{"task_id": "123"}`
2.  **`/ap/v1/agent/tasks/{task_id}/steps` (POST):** Execute the next step.
    *   Output: `{"step_id": "456", "status": "running", "output": "Searching flights..."}`
3.  **`/ap/v1/agent/tasks/{task_id}/artifacts` (GET):** List generated files (images, code).

### The Execution Loop

The Protocol enforces a **Step-by-Step** execution model.
1.  User creates Task.
2.  Client requests "Next Step".
3.  Agent performs *one* unit of work (thought + action).
4.  Agent returns status (`running` or `completed`) and artifacts.
5.  Client repeats until status is `completed`.

### Why this matters

1.  **Universal UIs:** You can build a frontend (like "Agent-UI") that works with *any* backend agent (AutoGPT, LangChain, custom) as long as they speak the protocol.
2.  **Benchmarking:** Benchmarks like **AgentBench** use this protocol to evaluate agents. They send a task, loop through steps, and check the artifacts.
3.  **Interoperability:** An "Orchestrator Agent" can spawn a "Sub-Agent" via a standard HTTP request, without needing to import the sub-agent's code.

### Comparison: MCP vs Agent Protocol

*   **MCP:** Internal. "How the Agent talks to its tools." (Southbound).
*   **Agent Protocol:** External. "How the User/World talks to the Agent." (Northbound).

### Summary

The Agent Protocol turns Agents into **Microservices**. It abstracts away the internal loop (ReAct, Plan-and-Solve) and exposes a clean, stateful API for task execution.

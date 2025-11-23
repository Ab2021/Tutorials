# Day 62: Testing & Debugging Multi-Agent Systems
## Core Concepts & Theory

### The "Black Box" Multiplied

Debugging a single LLM is hard. Debugging 5 LLMs talking to each other asynchronously is a nightmare.
*   **Non-Determinism:** The same input might lead to different conversation flows.
*   **Emergent Bugs:** Agent A works, Agent B works, but together they get stuck in a loop.

### 1. Observability (Tracing)

You cannot debug what you cannot see.
*   **Trace:** A timeline of events. `User Input -> Agent A Thought -> Agent A Tool Call -> Agent B Reply`.
*   **Tools:** LangSmith, Arize Phoenix, AgentOps.
*   **Visualization:** Gantt charts or Sequence Diagrams are essential to understand the flow.

### 2. Unit Testing Agents

*   **Mocking Partners:** To test Agent A, you mock Agent B.
    *   *Test:* "If Agent B says 'Error', does Agent A retry?"
    *   *Mock:* Hardcode Agent B's response to "Error".
*   **Invariant Checks:** "The conversation should never exceed 10 turns." "The agent should never output a credit card number."

### 3. Integration Testing (The Swarm)

*   **Simulator:** Run the swarm in a controlled environment.
*   **Golden Scenarios:** "The Refund Flow".
    *   Input: "I want a refund."
    *   Expected: Refund Agent is called, DB is updated, User gets confirmation.
*   **Evaluator:** An LLM that reads the trace and grades it (Pass/Fail).

### 4. Common Multi-Agent Bugs

*   **Infinite Loops:** "Thank you" -> "You're welcome" -> "No, thank you".
*   **Handoff Drops:** Agent A hands off to B, but B doesn't receive the context (User ID).
*   **Role Drift:** The "Coder" starts acting like the "Manager".
*   **Deadlocks:** Agent A waits for B, B waits for A.

### 5. Chaos Engineering

*   **Fault Injection:** Deliberately make a tool fail. Does the swarm recover?
*   **Message Dropping:** What if a message is lost?
*   **Latency Injection:** What if Agent B takes 30 seconds to reply?

### Summary

Testing Multi-Agent Systems requires moving from "Vibe Checking" (looking at the chat) to **Structured Engineering**. We need traces, metrics, and automated regression tests to ship with confidence.

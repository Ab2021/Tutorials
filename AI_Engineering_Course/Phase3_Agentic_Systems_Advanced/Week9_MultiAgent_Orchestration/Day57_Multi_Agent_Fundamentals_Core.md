# Day 57: Multi-Agent Fundamentals
## Core Concepts & Theory

### Why Multiple Agents?

A single agent (like GPT-4) is a generalist. But for complex tasks, **Specialization** beats Generalization.
*   **Single Agent:** "Write code, test it, review it, and deploy it." (Context overload, hallucination).
*   **Multi-Agent:**
    *   *Coder:* "I only write code."
    *   *Tester:* "I only write tests."
    *   *Reviewer:* "I only check for bugs."
    *   *Manager:* "I coordinate everyone."

### 1. The Actor Model

Multi-agent systems (MAS) are inspired by the **Actor Model** in computer science.
*   **Independent:** Each agent has its own state and system prompt.
*   **Message Passing:** Agents communicate by sending messages (text/JSON) to each other.
*   **Asynchronous:** Agents can work in parallel.

### 2. Collaboration Patterns

How do agents work together?

*   **Sequential (Chain):** A -> B -> C. (e.g., Research -> Write -> Edit).
*   **Hierarchical (Manager-Worker):** A Manager breaks down the task and assigns sub-tasks to Workers (A, B, C).
*   **Joint (Swarm):** Agents talk to each other freely in a shared chat room until consensus is reached.

### 3. Collaboration vs. Competition

*   **Collaboration:** All agents share a common goal (e.g., "Build this app").
*   **Competition:** Agents have opposing goals (e.g., "Debate this topic").
    *   *Debate:* Agent A argues Pro, Agent B argues Con. A Judge Agent decides the winner. This often yields higher quality reasoning than a single agent trying to be "balanced".

### 4. Handoffs (The Router)

The critical component in MAS is the **Handoff**.
*   *Scenario:* The "Triage Agent" talks to the user. User asks for a refund.
*   *Action:* Triage Agent performs a **Handoff** to the "Refund Agent".
*   *Mechanism:* This involves transferring the *Conversation History* and *Control* to the new agent.

### 5. Shared State (Blackboard Pattern)

Agents need a way to share data without passing massive message logs.
*   **Blackboard:** A shared memory space (e.g., a JSON object or Database) that all agents can read/write.
*   *Example:* The "Architect" writes the file structure to the Blackboard. The "Coder" reads it and writes the code.

### 6. Challenges

*   **Infinite Loops:** Agents thanking each other forever.
*   **Cost:** 5 agents = 5x the token cost.
*   **Coordination Overhead:** The time spent "talking about work" vs "doing work".

### Summary

Multi-Agent Systems allow us to build **Cognitive Architectures** that mimic human organizations. By decomposing a complex problem into roles, we can achieve higher performance and reliability than a single monolithic prompt.

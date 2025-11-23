# Day 71: Agent-to-Agent (A2A) Communication Patterns
## Core Concepts & Theory

### Beyond the Chatbot

Most "Multi-Agent" demos today are just a single LLM pretending to be different people in a loop.
**True A2A** involves independent runtime processes (often on different machines) communicating to achieve a goal.
They don't just "chat"; they **Negotiate**, **Coordinate**, and **Transact**.

### 1. The Language of Agents

Agents need a structured language, not just natural language.
*   **FIPA-ACL (Foundation for Intelligent Physical Agents):** The grandfather of agent standards (1990s).
    *   *Performatives:* `REQUEST`, `INFORM`, `AGREE`, `REFUSE`, `PROPOSE`.
    *   *Content:* The payload (e.g., "Weather is sunny").
*   **KQML (Knowledge Query and Manipulation Language):** Another classic standard.
*   **Modern JSON-RPC:** The de-facto standard today. Simple, stateless, easy to parse.

### 2. Communication Topologies

*   **Direct (P2P):** Agent A talks to Agent B directly. Fast, but requires A to know B's address.
*   **Broker/Mediator:** Agent A talks to a Broker. Broker routes to B. Good for decoupling.
*   **Blackboard (Pub/Sub):** Agent A posts a message to a shared board. Agent B reads it. Good for swarms.

### 3. The "Handshake"

Before exchanging data, agents must establish context.
1.  **Discovery:** "Who are you?"
2.  **Capability Check:** "Can you handle PDF parsing?"
3.  **Ontology Alignment:** "When I say 'Bank', do I mean a river bank or a financial bank?" (Crucial for heterogeneous agents).

### 4. Asynchronous Messaging

Agents operate at different speeds.
*   **Mailbox Pattern:** Agent A sends a message to Agent B's inbox. Agent B processes it when free.
*   **Future/Promise:** Agent A sends a request and gets a "Ticket". It polls later for the result.

### 5. Error Handling in A2A

*   **Timeout:** Agent B died.
*   **Misunderstanding:** Agent B didn't understand the prompt.
*   **Refusal:** Agent B understood but refuses (policy violation, low payment).

### Summary

A2A communication is the "TCP/IP" of the Agentic Web. It defines the packet structure and the routing logic that allows a diverse ecosystem of agents to work together without a central controller.

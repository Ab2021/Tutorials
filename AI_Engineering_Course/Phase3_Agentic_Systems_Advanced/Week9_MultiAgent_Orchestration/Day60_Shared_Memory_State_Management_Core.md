# Day 60: Shared Memory & State Management in Swarms
## Core Concepts & Theory

### The "Silo" Problem

In a standard Multi-Agent System, Agent A doesn't know what Agent B is thinking. They only know what they *say* to each other.
*   **Inefficiency:** Agent B re-discovers facts Agent A already found.
*   **Inconsistency:** Agent A thinks `user_id=1`, Agent B thinks `user_id=2`.

**Shared Memory** solves this by providing a "Single Source of Truth".

### 1. Types of Shared Memory

*   **Short-Term (The Blackboard):** A JSON object or in-memory dict shared by the current session.
    *   *Usage:* Storing the current plan, code snippets, or test results.
*   **Long-Term (The Knowledge Base):** A Vector DB or SQL DB.
    *   *Usage:* Storing user preferences, past project artifacts.
*   **Episodic (The Log):** A chronological record of all actions taken by the swarm.

### 2. The Blackboard Pattern

A classic AI pattern.
*   **The Board:** A central repository of data.
*   **The Experts:** Agents who watch the board.
*   **The Protocol:**
    1.  Agent A writes a partial solution to the board.
    2.  Agent B sees the change, performs its specialty, and updates the board.
    3.  Agent C verifies the board.

### 3. State Management (Graph State)

In LangGraph, the "State" is the Blackboard.
*   It is a typed schema (Pydantic/TypedDict).
*   Every node receives the *current state* and returns a *state update*.
*   **Reducers:** Functions that define how to merge the update (Overwrite? Append?).

### 4. Semantic Shared Memory (Vector)

Agents can communicate via embeddings.
*   Agent A: "I found a relevant document." -> Embeds and saves to Vector DB.
*   Agent B: "I need info on X." -> Queries Vector DB.
*   *Benefit:* Decoupled communication. Agent A doesn't need to know Agent B exists.

### 5. Concurrency & Locking

If two agents write to the Blackboard simultaneously, who wins?
*   **Optimistic Locking:** Last write wins (simple).
*   **Transactional:** Agents must "checkout" a section of the state, modify it, and "commit" it.
*   **Append-Only:** Agents never overwrite; they only add new entries (Event Sourcing).

### Summary

Shared Memory transforms a group of chatting agents into a **Coordinated Swarm**. It allows for implicit communication ("Stigmergy")—communicating by modifying the environment—which is how ants and bees coordinate.

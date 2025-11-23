# Day 48: Cognitive Architectures (Generative Agents)
## Core Concepts & Theory

### What is a Cognitive Architecture?

A blueprint for an intelligent agent that mimics the human mind.
It organizes **Memory**, **Perception**, **Planning**, and **Action** into a coherent system.
Standard ReAct is a simple architecture. We can build more complex ones.

### 1. Generative Agents (Park et al., 2023)

The "Sims" paper. 25 agents lived in a digital village.
**Key Components:**
1.  **Memory Stream:** A log of *everything* the agent has experienced.
2.  **Retrieval:** Fetching relevant memories based on Recency, Importance, and Relevance.
3.  **Reflection:** Synthesizing low-level memories into high-level insights ("I am friends with Alice").
4.  **Planning:** Creating a daily schedule and reacting to events.

### 2. MemGPT (Memory-GPT)

Treats the LLM as an OS.
*   **Main Context:** RAM (Limited).
*   **External Storage:** Disk (Unlimited Vector DB).
*   **Paging:** The agent autonomously moves information between RAM and Disk.
*   **Events:** System interrupts (User message, Timer).

### 3. BabyAGI / AutoGPT

**Loop:**
1.  **Task List:** Maintain a prioritized list of tasks.
2.  **Execution:** Pop top task. Execute.
3.  **Creation:** Based on result, create new tasks.
4.  **Prioritization:** Re-order the list.

### 4. SOAR / ACT-R

Classical cognitive architectures (pre-LLM).
They focused on "Production Rules" (If X then Y).
Modern architectures replace the rigid rules with the flexible reasoning of LLMs.

### Summary

A Cognitive Architecture gives the LLM **Persistence** and **Personality**. It turns a stateless text predictor into a long-lived digital entity.

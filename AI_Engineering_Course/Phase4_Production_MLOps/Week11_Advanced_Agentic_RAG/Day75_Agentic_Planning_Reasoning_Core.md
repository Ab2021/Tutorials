# Day 75: Agentic Planning & Reasoning
## Core Concepts & Theory

### From Reacting to Planning

**ReAct:** Good for short tasks.
**Planning:** Essential for long-horizon tasks ("Write a novel", "Plan a vacation").
**Goal:** Break a complex goal into manageable sub-tasks.

### 1. Chain of Thought (CoT)

**Concept:**
- Prompting the model to "Think step by step".
- **Mechanism:** Generating intermediate reasoning tokens improves final answer accuracy.
- **Zero-Shot CoT:** "Let's think step by step."
- **Few-Shot CoT:** Providing examples of reasoning.

### 2. Tree of Thoughts (ToT)

**Concept:**
- Explore multiple reasoning paths simultaneously.
- **Tree:** Root = Problem. Branches = Possible steps.
- **Search:** BFS or DFS to find the best path.
- **Evaluation:** Self-evaluate each state ("Is this path promising?").

### 3. ReAct (Reason + Act)

**Concept:**
- Loop: `Thought -> Action -> Observation -> Thought`.
- **Benefit:** Grounding. The model sees the result of its action before planning the next step.
- **Limitation:** Can get stuck in local loops.

### 4. Plan-and-Solve (PS)

**Concept:**
- **Phase 1:** Generate a complete plan (Step 1, Step 2, Step 3).
- **Phase 2:** Execute the plan.
- **Benefit:** Better global coherence than ReAct.
- **Limitation:** If Step 1 fails, the whole plan might be invalid.

### 5. Reflexion (Self-Correction)

**Concept:**
- Agent tries a task. Fails.
- **Reflect:** "Why did I fail?"
- **Memory:** Store the reflection.
- **Retry:** Try again, using the reflection to avoid the mistake.

### 6. LLM Compiler (Parallel Function Calling)

**Concept:**
- If a plan has independent steps ("Search for weather in NY", "Search for weather in London"), execute them in parallel.
- **DAG:** Directed Acyclic Graph of tasks.
- **Benefit:** Massive latency reduction.

### 7. Reasoning Models (o1 / Strawberry)

**Concept:**
- Models trained specifically to "think" for a long time before answering.
- **Inference-Time Compute:** Trading time for accuracy.
- **Internal Chain of Thought:** The model generates hidden reasoning tokens.

### 8. Decomposition

**Concept:**
- Breaking a problem down.
- **Recursive:** Break sub-tasks into sub-sub-tasks.
- **Orchestrator:** Assigns sub-tasks to workers.

### 9. Summary

**Planning Strategy:**
1.  **Simple:** Use **CoT** or **ReAct**.
2.  **Complex:** Use **Plan-and-Solve** or **Tree of Thoughts**.
3.  **Parallel:** Use **LLM Compiler** for independent tasks.
4.  **Improvement:** Use **Reflexion** to learn from failure.
5.  **Model:** Use **Reasoning Models (o1)** for hard logic.

### Next Steps
In the Deep Dive, we will implement a Tree of Thoughts search, a Reflexion loop, and a Plan-and-Solve agent.

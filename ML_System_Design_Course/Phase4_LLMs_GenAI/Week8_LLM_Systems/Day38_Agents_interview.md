# Day 38: Agents - Interview Questions

> **Topic**: Autonomous AI
> **Focus**: 20 Popular Interview Questions with Detailed Answers

### 1. What is an AI Agent?
**Answer:**
*   LLM + Memory + Tools + Planning.
*   Can perceive environment, reason, act, and observe results.

### 2. Explain the ReAct (Reason + Act) prompting strategy.
**Answer:**
*   Loop: **Thought** (Reasoning) -> **Action** (Tool Call) -> **Observation** (Tool Output).
*   Allows model to correct itself and handle multi-step tasks.

### 3. What is "Chain of Thought" (CoT)?
**Answer:**
*   Prompting model to "Think step by step".
*   Elicits reasoning. Improves performance on logic/math.

### 4. What is "Tool Use" (Function Calling)?
**Answer:**
*   Model outputs a structured call: `get_weather(city="London")`.
*   System executes code. Returns result to model.
*   Model uses result to generate answer.

### 5. Explain "Tree of Thoughts" (ToT).
**Answer:**
*   Generalization of CoT.
*   Explore multiple reasoning paths (Tree search: BFS/DFS).
*   Self-evaluate states. Backtrack if needed.

### 6. What is "Memory" in Agents?
**Answer:**
*   **Short-term**: Context window (Conversation history).
*   **Long-term**: Vector Database (Retrieval).

### 7. What is "Reflection" (Self-Correction)?
**Answer:**
*   Agent critiques its own past actions.
*   "I failed to find the file. I should try a different search query."
*   Adds robustness.

### 8. What is AutoGPT / BabyAGI?
**Answer:**
*   Autonomous loops.
*   Given a goal, agent breaks it down into tasks, executes them, adds new tasks, until goal is met.

### 9. What are the risks of Agents?
**Answer:**
*   **Infinite Loops**: Getting stuck.
*   **Hallucination**: Calling non-existent tools.
*   **Security**: Prompt Injection leading to dangerous actions (e.g., `delete_files`).

### 10. How do you evaluate Agents?
**Answer:**
*   **Success Rate**: Did it achieve the goal?
*   **Steps taken**: Efficiency.
*   **AgentBench / GAIA**.

### 11. What is "Plan-and-Solve"?
**Answer:**
*   Generate a full plan first. Then execute.
*   Contrast with ReAct (Interleaved planning and execution).

### 12. What is "Multi-Agent" collaboration?
**Answer:**
*   Role-playing.
*   **Coder** Agent writes code. **Reviewer** Agent checks it. **Manager** Agent coordinates.
*   Often outperforms single agent.

### 13. How do you handle "Context Limit" in long-running agents?
**Answer:**
*   **Summarization**: Compress old history.
*   **Retrieval**: Store history in Vector DB, retrieve relevant parts.
*   **FIFO**: Drop oldest messages.

### 14. What is "RAG" vs "Agent"?
**Answer:**
*   **RAG**: Read-only. Retrieve and Answer.
*   **Agent**: Read-Write. Can take actions (API calls, DB writes).

### 15. What is "Prompt Engineering" for Agents?
**Answer:**
*   Defining tools clearly.
*   Providing few-shot examples of tool usage.
*   System prompt: "You are a helpful assistant..."

### 16. What is "Symbolic AI" integration?
**Answer:**
*   LLM is bad at math. Agent delegates math to a Calculator tool (Python/Wolfram).
*   Neuro-symbolic approach.

### 17. What is the "Orchestrator" pattern?
**Answer:**
*   Central LLM decides which sub-agent or tool to call.
*   Router.

### 18. How do you debug an Agent?
**Answer:**
*   Trace the execution (LangSmith).
*   Inspect Thought/Action/Observation logs.

### 19. What is "Human-in-the-loop"?
**Answer:**
*   Agent asks for permission before critical actions (e.g., sending email).
*   Agent asks for clarification if stuck.

### 20. What is the future of Agents?
**Answer:**
*   OS-level agents (controlling computer).
*   Multimodal agents (seeing screen).
*   Standardized Agent Protocols.

# Day 45: ReAct Pattern Deep Dive (Reasoning + Acting)
## Core Concepts & Theory

### The Missing Link

CoT allows reasoning (Thought).
Tools allow action (Act).
**ReAct** (Yao et al., 2022) combines them: **Re**asoning + **Act**ing.

### 1. The Loop

Instead of just generating an answer, the model generates a trace:
1.  **Thought:** "I need to find out who the CEO of Apple is."
2.  **Action:** `Search("Apple CEO")`
3.  **Observation:** (From Tool) "Tim Cook is the CEO..."
4.  **Thought:** "Now I need to find his age."
5.  **Action:** `Search("Tim Cook age")`
6.  **Observation:** "63 years old."
7.  **Thought:** "I have the answer."
8.  **Answer:** "Tim Cook is 63."

### 2. Why it works

*   **Grounding:** The reasoning is grounded in external reality (Observations), reducing hallucination.
*   **Correction:** If the Action fails (Observation: "No results"), the model can Reason ("Maybe I spelled it wrong") and retry.
*   **Interpretability:** You can see exactly *why* the agent did what it did.

### 3. Comparison with other methods

*   **Act-Only:** (WebGPT) Just clicks links. Hard to plan long-term.
*   **Reason-Only:** (CoT) Just thinks. Hallucinates facts it doesn't know.
*   **ReAct:** Best of both.

### 4. Challenges

*   **Error Propagation:** One bad observation can derail the whole chain.
*   **Looping:** "Search -> Fail -> Search -> Fail". The agent gets stuck.
*   **Context Length:** The trace (Thought/Action/Observation) grows quickly, filling the window.

### Summary

ReAct is the standard architecture for "Agents" today (LangChain Agents, AutoGPT). It turns the LLM into a **Controller** for external tools.

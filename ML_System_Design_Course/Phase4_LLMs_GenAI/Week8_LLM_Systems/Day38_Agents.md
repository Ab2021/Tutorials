# Day 38: Agents & Tool Use

> **Phase**: 4 - LLMs & GenAI
> **Week**: 8 - LLM Systems
> **Focus**: From Chatbots to Do-ers
> **Reading Time**: 50 mins

---

## 1. The Agentic Workflow

An LLM is a brain in a jar. Agents give it hands (Tools).

### 1.1 ReAct (Reason + Act)
*   **Loop**:
    1.  **Thought**: "I need to find the weather in Tokyo."
    2.  **Action**: `get_weather("Tokyo")`
    3.  **Observation**: "25°C, Sunny"
    4.  **Thought**: "The weather is 25°C. I can answer now."
    5.  **Answer**: "It is 25°C in Tokyo."

### 1.2 Tool Calling (Function Calling)
*   **Fine-tuning**: Models (GPT-4, Llama 3) are fine-tuned to output structured JSON `{ "function": "get_weather", "args": {...} }` instead of text when they need a tool.

---

## 2. Architectures

### 2.1 Single Agent
*   Has a list of tools (Search, Calculator, Database).
*   Decides which to use.

### 2.2 Multi-Agent (AutoGen / CrewAI)
*   **Specialization**:
    *   *Coder Agent*: Writes code.
    *   *Reviewer Agent*: Checks code.
    *   *Manager Agent*: Coordinates.
*   **Benefit**: Specialized prompts/tools for each role reduce hallucination and improve complex task completion.

---

## 3. Real-World Challenges & Solutions

### Challenge 1: Infinite Loops
**Scenario**: Agent tries `search("Python")`, fails, tries `search("Python")` again. Loops forever.
**Solution**:
*   **Max Iterations**: Hard limit (e.g., 10 steps).
*   **History Check**: If Action is identical to previous Action, force stop or change strategy.

### Challenge 2: Context Overflow
**Scenario**: Agent browses the web. Scrapes a 50-page PDF. Context window full.
**Solution**:
*   **Summarization**: Summarize observation before adding to history.
*   **Memory Bank**: Store observations in Vector DB, retrieve only relevant ones.

---

## 4. Interview Preparation

### Conceptual Questions

**Q1: What is the difference between Chain-of-Thought (CoT) and ReAct?**
> **Answer**:
> *   **CoT**: Internal reasoning only. "Let's think step by step." No external actions.
> *   **ReAct**: Reasoning + External Actions. It interacts with the world to get new information.

**Q2: How do you prevent an Agent from doing dangerous things (SQL Injection)?**
> **Answer**:
> 1.  **Read-Only Permissions**: Give the DB tool read-only access.
> 2.  **Human-in-the-loop**: Require user approval for sensitive actions (e.g., `delete_file`).
> 3.  **Sandboxing**: Run code execution tools in a secure Docker container (e2b).

**Q3: Why do Agents often get stuck in loops?**
> **Answer**: LLMs are stateless. If the error message from a tool is not informative ("Error 500"), the LLM doesn't know *why* it failed, so it retries the same thing hoping for a different result. Better error messages help.

---

## 5. Further Reading
- [ReAct Paper](https://arxiv.org/abs/2210.03629)
- [AutoGen: Enabling Next-Gen LLM Applications](https://microsoft.github.io/autogen/)

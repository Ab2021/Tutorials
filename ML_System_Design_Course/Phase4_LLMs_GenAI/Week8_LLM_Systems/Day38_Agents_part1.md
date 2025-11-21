# Day 38 (Part 1): Advanced Agents

> **Phase**: 6 - Deep Dive
> **Topic**: Autonomous AI
> **Focus**: ReAct, Planning, and Tool Use
> **Reading Time**: 60 mins

---

## 1. ReAct Internals

Reasoning + Acting.

### 1.1 The Loop
1.  **Thought**: "I need to find the weather."
2.  **Action**: `get_weather("London")`.
3.  **Observation**: "15C, Rainy".
4.  **Thought**: "It is raining. I should bring an umbrella."
5.  **Action**: `Finish("Bring umbrella")`.

---

## 2. Planning Strategies

### 2.1 Plan-and-Solve (Least-to-Most)
*   **Problem**: ReAct gets lost in long chains.
*   **Fix**: Generate a full plan *first*.
    1.  Search for X.
    2.  Calculate Y.
    3.  Compare.
*   Execute step-by-step.

---

## 3. Tricky Interview Questions

### Q1: How to prevent Infinite Loops in Agents?
> **Answer**:
> *   **Max Steps**: Hard limit (e.g., 10 steps).
> *   **History Check**: If Action/Observation is identical to previous step, force stop or temperature increase.
> *   **Timeout**: Wall clock limit.

### Q2: Function Calling (OpenAI) vs ReAct?
> **Answer**:
> *   **ReAct**: Prompt engineering. Model outputs text "Action: ...". Fragile parsing.
> *   **Function Calling**: Fine-tuned capability. Model outputs structured JSON. Robust.

### Q3: Memory: Short-term vs Long-term?
> **Answer**:
> *   **Short**: Context Window (Messages list).
> *   **Long**: Vector DB (Reflection).
> *   **Generative Agents**: Retrieve memories -> Summarize -> Store summary.

---

## 4. Practical Edge Case: Hallucinated Tools
*   **Problem**: Model calls `get_stock_price("AAPL")` but tool is named `get_ticker_price`.
*   **Fix**: RAG on Tool Definitions. Retrieve relevant tool schemas before prompting.


# Day 51: Building Custom Tools (LangChain & LlamaIndex)
## Core Concepts & Theory

### Beyond the Basics

Yesterday we built raw tool executors. Today, we look at how frameworks like **LangChain** and **LlamaIndex** abstract this process. While you can build everything from scratch, these frameworks provide utilities for:
1.  **Standardized Interfaces:** A common `BaseTool` class.
2.  **Output Parsing:** Handling the messy regex/JSON parsing automatically.
3.  **Pre-built Toolkits:** Access to Gmail, Google Search, SQL, etc., out of the box.

### 1. LangChain Tools

In LangChain, a Tool consists of:
*   **Name:** How the LLM calls it.
*   **Description:** When the LLM should call it.
*   **Args Schema:** A Pydantic model defining inputs.
*   **Run Function:** The actual logic.

**The `@tool` Decorator:**
The easiest way to define a tool. It uses the function's name, docstring, and type hints to auto-generate the schema.

```python
@tool
def search(query: str) -> str:
    """Search the web for the query."""
    return "results..."
```

### 2. Structured Tools

For complex inputs (multiple arguments), LangChain uses `StructuredTool`. This enforces the schema validation we implemented manually yesterday. It ensures that if an agent tries to call a tool with missing arguments, it raises a `ValidationError` which LangChain's agent executor catches and feeds back to the model.

### 3. LlamaIndex Data Agents

LlamaIndex focuses heavily on **Data Tools**.
*   **QueryEngineTool:** Wraps a RAG pipeline as a tool. This is powerful. It allows an agent to "consult the documentation" as just one step in a larger plan.
*   **FunctionTool:** Similar to LangChain's `@tool`.

### 4. Toolkits

A **Toolkit** is a collection of related tools designed to be used together.
*   **SQL Toolkit:** Tools to `list_tables`, `describe_table`, `execute_sql`.
*   **GitHub Toolkit:** Tools to `get_issue`, `create_pr`, `read_file`.
*   **Office365 Toolkit:** Email, Calendar, OneDrive.

Using a toolkit is often better than individual tools because the tools share context (e.g., an authenticated client session).

### 5. The "Agent Executor" Loop

LangChain's `AgentExecutor` is a pre-built loop that handles:
1.  **Thought:** LLM decides what to do.
2.  **Action:** LLM selects a tool.
3.  **Action Input:** LLM provides arguments.
4.  **Observation:** Tool executes and returns result.
5.  **Repeat:** Until the LLM produces a "Final Answer".

**ReAct vs. OpenAI Functions Agent:**
*   **ReAct:** Uses text-based prompting (`Thought: ... Action: ...`). Works on any model.
*   **OpenAI Functions:** Uses the native API. More robust, less parsing errors.

### 6. Dynamic Tool Loading

In production, you might have user-specific tools.
*   User A has connected their Gmail.
*   User B has connected their Jira.
*   **Context Injection:** When the agent starts, you check the user's permissions and load only the relevant toolkits into the agent's `tools` list.

### 7. Handling Large Outputs

A common issue with custom tools (e.g., `read_file` or `sql_query`) is returning too much data.
*   **Truncation:** Automatically cut off after N tokens.
*   **Summarization Chain:** If output > N tokens, trigger a sub-call to summarize it before returning to the main agent.
*   **Error Handling:** If a SQL query returns 1M rows, the tool should catch this and return "Too many results, please refine query" instead of crashing the context window.

### Summary

Frameworks like LangChain and LlamaIndex standardize the "plumbing" of tool use. They allow you to focus on the *logic* of your tools rather than the *parsing* of JSON. The most powerful pattern is wrapping complex RAG pipelines as simple tools, giving your agent "expert" capabilities.

# Day 51: Building Custom Tools (LangChain & LlamaIndex)
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does the `AgentExecutor` in LangChain decide when to stop?

**Answer:**
The `AgentExecutor` runs a loop:
1.  It sends the history + user input to the LLM.
2.  The LLM returns either a **Tool Call** or a **Final Answer**.
3.  If it's a Tool Call, the executor runs the tool and appends the output to history. It then loops back to step 1.
4.  If it's a Final Answer, the loop breaks and returns the text to the user.
*   **Safety Valve:** There is also a `max_iterations` (or `max_execution_time`) parameter to prevent infinite loops if the model keeps calling tools without resolving the query.

#### Q2: What is the difference between `QueryEngineTool` and `FunctionTool` in LlamaIndex?

**Answer:**
*   **FunctionTool:** Wraps a standard Python function (e.g., `multiply(a, b)`). It's for general logic.
*   **QueryEngineTool:** Wraps a LlamaIndex `QueryEngine` (which wraps a Retriever + LLM). It's specifically for **RAG**.
*   **Why it matters:** A `QueryEngineTool` allows hierarchical reasoning. The main agent delegates a complex question ("Summarize the report") to the Query Engine, which does its own retrieval and synthesis, and returns a summary string to the main agent.

#### Q3: Why is "Output Parsing" a common point of failure in Agents?

**Answer:**
LLMs are probabilistic. Even with "JSON Mode", they might output:
*   Markdown code blocks: ` ```json {...} ``` `
*   Trailing commas (invalid JSON).
*   Comments inside JSON.
**Solution:** Robust parsers (like LangChain's `PydanticOutputParser`) use regex to strip markdown and `json.loads` with error recovery (e.g., trying to fix common syntax errors or asking the LLM to fix its own output).

#### Q4: How do you unit test a Custom Tool?

**Answer:**
You test the **logic**, not the LLM.
1.  **Mock the External API:** If your tool calls an API, use `unittest.mock` to mock the response.
2.  **Test the Run Function:** Call the tool's python function directly with various inputs.
3.  **Test Schema:** Verify that `tool.args_schema` correctly validates invalid inputs (e.g., passing a string to an integer field).

### Production Challenges

#### Challenge 1: The "Black Box" Agent

**Scenario:** The agent gives a wrong answer. You don't know if it was a bad retrieval, a bad tool call, or a bad synthesis.
**Root Cause:** Lack of observability.
**Solution:**
*   **Tracing:** Use **LangSmith** or **Arize Phoenix**.
*   **Log Everything:** Log the `intermediate_steps` in LangChain. You need to see: `Input -> Thought -> Tool Call -> Tool Output -> Final Answer`.

#### Challenge 2: Tool Argument Complexity

**Scenario:** You have a tool `create_jira_ticket` with 20 fields. The LLM constantly misses required fields or hallucinates values.
**Root Cause:** Cognitive overload for the model.
**Solution:**
*   **Simplify:** Break it into `create_basic_ticket(summary)` and `update_ticket_details(id, details)`.
*   **Defaults:** Make most fields optional in the schema with sensible defaults.
*   **Interactive:** If a field is missing, the tool should return "Error: Missing field X. Please ask the user for X."

#### Challenge 3: Authentication Management

**Scenario:** You build a GitHub tool. It works locally with your Personal Access Token (PAT). In production, User A should not be able to see User B's repos.
**Root Cause:** Shared credentials.
**Solution:**
*   **User-Scoped Tools:** Do not initialize tools globally. Initialize them **per request**.
*   **OAuth:** On request start, fetch User A's OAuth token from the session and pass it to the `GitHubTool` constructor.

#### Challenge 4: Infinite Loops / "I'm sorry" Loops

**Scenario:**
Agent: Calls `search("foo")`.
Tool: Returns "Error: API down".
Agent: Calls `search("foo")` again.
... Repeat until token limit.
**Root Cause:** The model thinks retrying the exact same thing will fix it.
**Solution:**
*   **System Prompt:** "If a tool fails, try a different strategy or ask the user. Do not retry the same action more than once."
*   **Tool Output:** Return "Error: API down. Do not retry."

### System Design Scenario: Enterprise Search Agent

**Requirement:** An agent that can search Wiki (Public), Jira (Internal), and Salesforce (Sales Data).
**Design:**
1.  **Tools:** `WikiTool`, `JiraTool`, `SalesforceTool`.
2.  **Router:** A top-level classifier (or the Agent itself) decides which tool to use.
3.  **Auth:**
    *   Wiki: No auth.
    *   Jira: OAuth (User context).
    *   Salesforce: Service Account (Read-only) or OAuth.
4.  **Performance:** Run searches in parallel if the user query implies multiple sources ("Check Jira and Salesforce for client X").

### Summary Checklist for Production
*   [ ] **Tracing:** Enable LangSmith/Phoenix.
*   [ ] **Timeouts:** Set global timeouts for the Agent Executor.
*   [ ] **Auth:** Ensure tools use user-scoped credentials.
*   [ ] **Fallbacks:** Handle tool failures gracefully (don't crash the app).
*   [ ] **Testing:** Unit test the tool logic independent of the LLM.

# Day 65: Building MCP Servers
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle "Long-Running Tasks" in MCP?

**Answer:**
Standard MCP tools are synchronous (Request -> Response). If a task takes 5 minutes:
1.  **Async Pattern:** The Tool starts the job and returns a `job_id` immediately.
2.  **Notification:** The Server sends a `notifications/message` when the job is done (if supported).
3.  **Polling:** The Agent calls a `check_status(job_id)` tool periodically.
*   *Note:* JSON-RPC has a timeout. Don't block the thread for 5 minutes.

#### Q2: What is the difference between `FastMCP` and the low-level `Server` class in Python?

**Answer:**
*   **FastMCP:** A high-level wrapper (like FastAPI). It uses decorators (`@mcp.tool()`) and handles the lifecycle, error catching, and type conversion automatically. Recommended for most users.
*   **Server:** The low-level class. You have to manually register handlers and manage the request loop. Useful if you need custom middleware or non-standard transport.

#### Q3: How do you version an MCP Server?

**Answer:**
*   **Protocol Version:** Negotiated during the `initialize` handshake.
*   **Server Version:** Sent in the `serverInfo` field.
*   **Tool Versioning:** If you change a tool's schema (breaking change), rename it (`get_weather_v2`) or use a new server version. The LLM adapts to the new schema automatically via In-Context Learning, but hardcoded clients might break.

#### Q4: Can an MCP Server call *another* MCP Server?

**Answer:**
Technically, yes, if the Server acts as a Client to another Server.
*   *Pattern:* A "Meta-Server" that aggregates multiple downstream servers.
*   *Use Case:* A "DevOps Server" that talks to "AWS Server" and "GitHub Server" to perform complex workflows.

### Production Challenges

#### Challenge 1: Schema Hallucination

**Scenario:** You define a tool with `unit: "C" | "F"`. The LLM sends `unit: "Celsius"`. The server crashes.
**Root Cause:** The LLM didn't respect the Enum strictly.
**Solution:**
*   **Lenient Parsing:** In your tool logic, normalize inputs. `if unit.lower().startswith("c"): return "C"`.
*   **Retry Error:** Return a specific error: "Invalid unit 'Celsius'. Please use 'C' or 'F'". The LLM will self-correct in the next turn.

#### Challenge 2: Rate Limiting

**Scenario:** The LLM enters a loop and calls your `send_email` tool 100 times in 1 minute.
**Root Cause:** Agentic loop gone wrong.
**Solution:**
*   **Token Bucket:** Implement a rate limiter inside the MCP Server.
*   **Cost Tracking:** Track the number of calls per session. Reject calls after a threshold.

#### Challenge 3: Dependency Conflicts

**Scenario:** You want to run 5 Python MCP servers. One needs `pandas==1.0`, another needs `pandas==2.0`.
**Root Cause:** Shared Python environment.
**Solution:**
*   **uv / venv:** Each MCP Server should be a standalone executable or run in its own virtual environment.
*   **Config:** `args: ["/path/to/venv1/python", "server1.py"]`.

### System Design Scenario: Database Analyst Server

**Requirement:** Allow Claude to query a Production Read-Replica DB.
**Design:**
1.  **Schema Discovery:** A resource `db://schema` that returns the `CREATE TABLE` statements. This fits into the context window better than raw data.
2.  **Query Tool:** `run_sql(query)`.
3.  **Safety Layer:**
    *   Parse SQL with `sqlparse`.
    *   Reject `DROP`, `ALTER`, `INSERT`.
    *   Enforce `LIMIT 100` on all SELECTs to prevent OOM.
4.  **Privacy:** Mask PII columns (SSN, Email) in the result set before returning to the LLM.

### Summary Checklist for Production
*   [ ] **Validation:** Use Pydantic/Zod for all inputs.
*   [ ] **Isolation:** Run servers in dedicated venvs or containers.
*   [ ] **Observability:** Log every tool call and result.
*   [ ] **Feedback:** Return helpful error messages, not stack traces.

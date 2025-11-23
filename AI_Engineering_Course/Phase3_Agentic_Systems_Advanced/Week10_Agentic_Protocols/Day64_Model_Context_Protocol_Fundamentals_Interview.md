# Day 64: Model Context Protocol (MCP) Fundamentals
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How does MCP differ from OpenAI's "GPT Actions" or LangChain Tools?

**Answer:**
*   **OpenAI/LangChain:** These are *framework-specific* or *platform-specific*. A LangChain tool cannot be natively used by LlamaIndex without a wrapper.
*   **MCP:** This is a *protocol* (standard). It operates at the process level (Stdio/HTTP). An MCP server is agnostic to the client. It just speaks JSON-RPC. It decouples the "Tool Logic" from the "Agent Framework".

#### Q2: Why use `stdio` transport instead of HTTP for local agents?

**Answer:**
*   **Security:** `stdio` (Standard Input/Output) is local-only. The server process is a child of the client. It is not exposed to the network/internet. No ports to open, no firewalls to configure.
*   **Simplicity:** No need to manage auth tokens (since the parent process owns the child).
*   **Latency:** Extremely low latency compared to HTTP overhead.

#### Q3: Explain the security model of MCP.

**Answer:**
MCP operates on **User Consent**.
*   The Server exposes capabilities.
*   The Client (Host) decides *if* and *when* to allow the LLM to use them.
*   Typically, "Read" operations (Resources) are allowed automatically, but "Write" operations (Tools) require **Human-in-the-Loop** confirmation (e.g., "Claude wants to run `execute_query`. Allow?").
*   The Server never sees the full conversation history; it only sees the specific tool call payload.

#### Q4: What are "Prompts" in MCP and why are they useful?

**Answer:**
Prompts are reusable context templates defined by the Server.
*   *Example:* A "Debugger Server" might define a prompt called `analyze-error`.
*   *Action:* When the user selects this prompt, the Server automatically bundles the relevant log files, the stack trace, and a system instruction ("You are a senior debugger...") and feeds it to the LLM.
*   *Benefit:* It captures "Domain Knowledge" on *how* to use the data, right next to the data itself.

### Production Challenges

#### Challenge 1: The "Context Window" Explosion

**Scenario:** You connect a "FileSystem MCP Server". The LLM decides to read 50 files. The context window overflows.
**Root Cause:** The Server provides raw data without limits.
**Solution:**
*   **Server-Side Truncation:** The Server should enforce a `max_bytes` limit on resource reads.
*   **Summarization Resource:** Instead of `read_file`, offer `summarize_file` which uses a cheap local LLM to compress the data before sending.

#### Challenge 2: Zombie Processes

**Scenario:** The Client crashes, but the MCP Server (subprocess) keeps running, holding a lock on the database.
**Root Cause:** Improper signal handling.
**Solution:**
*   **Heartbeats:** Implement JSON-RPC heartbeats. If the Client stops pinging, the Server should self-terminate.
*   **Parent-Death Signal:** On Linux/Mac, set `PR_SET_PDEATHSIG` so the child dies instantly if the parent dies.

#### Challenge 3: Auth Management for Remote Servers

**Scenario:** You want to connect to a remote "Salesforce MCP Server" over SSE (HTTP).
**Root Cause:** `stdio` doesn't work remotely.
**Solution:**
*   **OAuth flow:** The Host must handle the OAuth handshake with Salesforce, obtain the token, and pass it to the MCP Client, which sends it in the HTTP headers to the MCP Server.

### System Design Scenario: Enterprise Knowledge Graph

**Requirement:** Connect Claude to the company's internal Knowledge Graph (Neo4j), Jira, and Slack.
**Design:**
1.  **Servers:** Deploy 3 separate MCP Servers (Microservices pattern).
    *   `mcp-neo4j`
    *   `mcp-jira`
    *   `mcp-slack`
2.  **Deployment:** Run them as Docker containers in a private VPC.
3.  **Gateway:** Use an "MCP Gateway" (an SSE-to-Stdio bridge) that aggregates these 3 remote servers into a single endpoint for the local Client.
4.  **Security:** The Gateway handles SSO (Single Sign-On) and enforces Read-Only permissions for junior employees.

### Summary Checklist for Production
*   [ ] **Transport:** Use Stdio for local, SSE for remote.
*   [ ] **Validation:** Validate all Tool inputs on the Server side (Pydantic).
*   [ ] **Logging:** Log all JSON-RPC traffic for debugging.
*   [ ] **Timeouts:** Set strict timeouts for Tool execution (e.g., 30s).

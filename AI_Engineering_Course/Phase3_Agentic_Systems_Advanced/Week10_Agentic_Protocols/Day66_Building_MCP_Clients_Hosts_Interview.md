# Day 66: Building MCP Clients & Hosts
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why do we need to "Namespace" tools when aggregating multiple servers?

**Answer:**
Two servers might both define a tool called `read_file`.
*   Server A (Local FS): `read_file`
*   Server B (S3 Bucket): `read_file`
If you flatten them into one list, the LLM won't know which one to call, or the Client won't know which session to route the request to.
**Solution:** Rename them to `local_fs__read_file` and `s3__read_file`.

#### Q2: How does the Host handle "Sampling" (Context Compression)?

**Answer:**
When the Server sends a large resource, it might include a `sampling` hint.
The Host (Client) is responsible for:
1.  Checking the user's preference ("Allow sampling?").
2.  Calling an LLM (usually a smaller, cheaper one) to summarize the content.
3.  Passing the summary back to the Server (or using it directly).
This offloads the compute cost of summarization to the Host, which controls the model budget.

#### Q3: What happens if the MCP Server hangs?

**Answer:**
The Host must implement **Timeouts**.
*   If `session.call_tool()` takes > 60 seconds, the Host should kill the request and return a `TimeoutError` to the LLM.
*   The LLM can then decide to retry or apologize to the user.
*   Without timeouts, the entire UI freezes.

#### Q4: Explain the "Capabilities" handshake.

**Answer:**
During `initialize`, Client and Server exchange capabilities.
*   Client: "I support `sampling` and `roots` (file system roots)."
*   Server: "I support `resources`, `tools`, and `logging`."
This allows for backward compatibility. If the Client doesn't support `sampling`, the Server knows not to send sampling requests.

### Production Challenges

#### Challenge 1: The "Chatty" Protocol

**Scenario:** Listing tools from 10 servers takes 5 seconds on startup.
**Root Cause:** 10 sequential JSON-RPC calls.
**Solution:**
*   **Parallel Initialization:** Use `asyncio.gather` to initialize all sessions and fetch tool lists in parallel.
*   **Caching:** Cache the tool definitions on disk. Only re-fetch if the Server version changes.

#### Challenge 2: User Confirmation Fatigue

**Scenario:** The user has to click "Approve" for every single file read. They get annoyed and disable security.
**Root Cause:** Too granular permissions.
**Solution:**
*   **Session-Scoped Approval:** "Allow this Agent to read *any* file in `/project/src` for the next hour."
*   **Trust Levels:** Mark read-only tools as "Safe" (auto-approve) and write tools as "Sensitive" (require approval).

#### Challenge 3: Error Propagation

**Scenario:** The Server prints a Python stack trace to stderr. The Client crashes.
**Root Cause:** Unhandled stderr output.
**Solution:**
*   **Stderr Capture:** The Client should read the Server's `stderr` stream and log it to a debug console, *not* crash the main application.
*   **Log Messages:** The Server should use `notifications/message` to send logs to the Client properly, rather than printing to stderr.

### System Design Scenario: Cloud IDE with Agents

**Requirement:** An IDE (VS Code Web) where an Agent can access the user's containerized environment.
**Design:**
1.  **Host:** The Browser (running a WASM Client or communicating with a Backend Client).
2.  **Server:** A Node.js MCP Server running inside the user's Dev Container.
3.  **Transport:** WebSocket (bridging the Browser to the Container).
4.  **Flow:**
    *   User types "Fix the bug in main.py".
    *   Browser sends prompt to LLM.
    *   LLM calls `read_file`.
    *   Browser sends JSON-RPC over WebSocket to Container.
    *   Container reads file, returns content.
    *   Browser forwards to LLM.

### Summary Checklist for Production
*   [ ] **Concurrency:** Initialize servers in parallel.
*   [ ] **Namespacing:** Prefix tools to prevent collisions.
*   [ ] **Timeouts:** Enforce limits on all RPC calls.
*   [ ] **UX:** Build a clean UI for Tool Approval and Log viewing.

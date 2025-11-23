# Day 65: Building MCP Servers
## Core Concepts & Theory

### The Server Lifecycle

Building an MCP Server is similar to building a Web Server (FastAPI/Express), but simpler.
It involves:
1.  **Capabilities Declaration:** Telling the client "I support Resources and Tools".
2.  **Request Loop:** Listening for JSON-RPC messages.
3.  **Handlers:** Routing `tools/call` to a Python function.

### SDKs and Ecosystem

While you *can* write raw JSON-RPC over stdin, it's painful. Use the SDKs:
*   **Python SDK (`mcp`):** Great for data science, database connectors, and local automation.
*   **TypeScript SDK (`@modelcontextprotocol/sdk`):** Great for web integrations, browser automation, and Node.js apps.

### Designing Good Resources

A Resource is **Passive Data**.
*   **URI Scheme:** Define a clean scheme. `my-app://{id}/...`
*   **MIME Types:** Always specify the MIME type (`text/plain`, `application/json`, `image/png`). This helps the Client render it correctly.
*   **Updates:** If the data changes (e.g., a log file), use `notifications/resources/updated` to tell the Client to re-fetch.

### Designing Good Tools

A Tool is **Active Logic**.
*   **Atomic:** Tools should do one thing well. `add_user`, not `manage_system`.
*   **Descriptive:** The `description` field is the **Prompt** for the LLM. Be verbose. Explain edge cases.
*   **Schema:** Use strict JSON Schema (Pydantic in Python, Zod in TS). The LLM relies on this to generate correct JSON.

### Error Handling

When a Tool fails:
*   **Don't Crash:** Catch the exception.
*   **Return Error Text:** Return a structured string: `Error: User not found. Available users are: [Alice, Bob]`.
*   **Why?** This allows the Agent to *self-correct*. If you just throw a 500 error, the Agent gives up. If you explain the error, the Agent tries again with valid input.

### Security Considerations

*   **Input Validation:** Never trust the LLM. It might send `rm -rf /` as an argument.
*   **Read-Only Mode:** Consider a flag to run the server in "Read-Only" mode (Resources only, no Tools).
*   **Path Traversal:** If serving files, ensure the LLM can't request `file:///etc/passwd`.

### Summary

An MCP Server is a **Deterministic Wrapper** around a non-deterministic world. Your job is to expose the chaos of your internal APIs as a clean, structured interface that an LLM can understand and manipulate safely.

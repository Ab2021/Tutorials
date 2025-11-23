# Day 66: Building MCP Clients & Hosts
## Core Concepts & Theory

### The Role of the Host

While most people use the **Claude Desktop** app as their MCP Host, building a **Custom Host** is where the real power lies.
A Host is any application that:
1.  Spawns MCP Servers (subprocesses).
2.  Maintains the JSON-RPC connection.
3.  Aggregates Tools/Resources from all servers.
4.  Injects them into the LLM's context.
5.  Executes the tools when the LLM requests them.

### Client Architecture

An MCP Client typically has three layers:
1.  **Transport Layer:** Handles `stdin`/`stdout` or HTTP.
2.  **Protocol Layer:** Handles the JSON-RPC handshake (`initialize`, `ping`) and message routing.
3.  **Application Layer:** Merges the tools into a unified list for the LLM (e.g., OpenAI `tools` format).

### The "System Prompt" Injection

The magic of MCP is how it exposes tools to the LLM. The Client queries `tools/list` from the Server, converts them to the LLM's native format (e.g., OpenAI Function Definitions), and appends them to the API call.
*   **Resources:** The Client might eagerly fetch a resource and paste its content into the System Prompt if the user marked it as "Active".

### Security at the Host Level

The Host is the **Gatekeeper**.
*   **Consent UI:** When the LLM calls a tool, the Host intercepts the request. It shows a popup: "Agent wants to run `git push`. Allow?"
*   **Sandboxing:** The Host decides *how* to run the Server (e.g., inside a Docker container vs. bare metal).

### Multi-Server Aggregation

A Host can connect to 10 different servers.
*   Server A: Filesystem
*   Server B: GitHub
*   Server C: Postgres
The Host presents a "Super Tool List" to the LLM containing tools from A, B, and C. The LLM doesn't know they come from different processes. It just sees a toolbox.

### Summary

Building a Client allows you to embed Agentic capabilities into *your* app. Your IDE, your terminal, or your internal dashboard can become an Agent Host, leveraging the ecosystem of existing MCP Servers.

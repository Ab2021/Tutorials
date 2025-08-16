
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 1: Foundations of MCP

### Lesson 1.2: The MCP Architecture

### **Component Responsibilities: A Deep Dive**

---

### **1. Introduction: Defining Clear Roles**

In software architecture, the principle of **Separation of Concerns** states that a system should be divided into distinct sections, each addressing a separate concern. The MCP architecture embodies this principle through its three core components: the Host, the Client, and the Server. 

While we have introduced these components, a deeper, more granular understanding of their specific responsibilities is essential for any developer building an MCP application. This guide provides a detailed checklist of the duties for each component, clarifying who is responsible for what in the MCP ecosystem. Understanding these roles prevents confusion and helps you design cleaner, more maintainable systems.

---

### **2. The Host: The Master of the User Experience**

The Host is the world the user lives in. Its primary directive is to serve the user, not the protocol. MCP is a tool the Host uses to become more powerful, but its core responsibilities are to the user.

**Checklist of Host Responsibilities:**

*   **[ ] Render the User Interface (UI):**
    *   Displaying all visual elements: windows, buttons, text editors, chat panels, menus, etc.
    *   Managing the layout and theme of the application.
    *   **Example:** In an IDE, the Host is responsible for the entire window, including the file tree, the code editor, and the terminal panel.

*   **[ ] Handle User Input:**
    *   Capturing keyboard strokes, mouse clicks, drag-and-drop events, and voice commands.
    *   Translating these low-level events into high-level application commands.
    *   **Example:** The Host detects that the user has typed `/commit` into a chat box and decides to initiate an MCP prompt.

*   **[ ] Manage Application State:**
    *   Knowing which user is logged in.
    *   Knowing which files are currently open or which database is connected.
    *   Keeping track of the user's selection or the current state of a multi-step workflow.
    *   **Example:** The Host knows that the file `main.py` is the currently active document, so it can tell the Client to make `file:///path/to/main.py` available as a contextual resource.

*   **[ ] Launch and Configure MCP Servers:**
    *   Deciding which MCP servers are needed for the application's functionality.
    *   Knowing the path to local server executables.
    *   Containing the logic to launch these executables as subprocesses.
    *   **Example:** An IDE Host knows that for a Python project, it should launch the `python-linter-mcp-server`, and for a Git repository, it should launch the `git-mcp-server`.

*   **[ ] Own and Manage the MCP Client:**
    *   The MCP Client is a component *within* the Host. The Host is responsible for creating the client instance, providing it with the necessary configuration (like the server commands), and managing its lifecycle.

*   **[ ] Mediate User-Facing Security and Permissions:**
    *   When the Client reports that a server wants to perform a dangerous action, it is the Host's responsibility to render the confirmation dialog to the user.
    *   The Host must provide the UI for the user to grant or deny permission.
    *   **Example:** The Host displays a modal dialog: "The `git-server` wants to delete the branch `feature-x`. [Allow] [Deny]".

---

### **3. The MCP Client: The Protocol Engine and Security Guard**

The Client is the heart of the MCP communication. It is the user's trusted agent, handling all the details of the protocol and enforcing security boundaries.

**Checklist of Client Responsibilities:**

*   **[ ] Manage Server Lifecycle:**
    *   Executing the command provided by the Host to start the server process.
    *   Establishing the transport connection (e.g., connecting to the server's `stdio` pipes).
    *   Terminating the server process when the connection is no longer needed.

*   **[ ] Speak JSON-RPC 2.0:**
    *   Constructing and serializing valid JSON-RPC Request objects.
    *   Sending these requests over the chosen transport.
    *   Receiving and deserializing JSON-RPC Response and Notification objects.
    *   Matching response `id`s to request `id`s to correlate asynchronous messages.

*   **[ ] Perform the `initialize` Handshake:**
    *   Sending the initial `initialize` request upon establishing a new connection.
    *   Passing the `rootUris` provided by the Host to the server to set the security sandbox.
    *   Verifying that the connected server's name and version match expectations.

*   **[ ] Discover and Expose Capabilities:**
    *   Calling `prompts/list`, `resources/list`, and `tools/list` to discover what the server can do.
    *   Making this list of capabilities available to the Host (so it can build UIs) and potentially to the LLM (so it can decide which tools to call).

*   **[ ] Orchestrate LLM Interactions:**
    *   Taking a user's query and the list of available tools and formatting them into a prompt for the LLM.
    *   Sending the prompt to the configured LLM API.
    *   Receiving the LLM's response, which could be a natural language answer or a `tool_call` request.
    *   Executing the `tool_call` by sending a `tools/call` request to the appropriate MCP server.
    *   Taking the tool's result and sending it back to the LLM for final synthesis.

*   **[ ] Enforce Security Policies:**
    *   Acting as the gatekeeper for all communication.
    *   Refusing to send requests for resources that are outside the configured `rootUris`.
    *   Intercepting `tool_call` requests for tools that are marked as dangerous or destructive.
    *   Notifying the Host that user permission is required before proceeding with a dangerous action.

*   **[ ] Handle Server-Side Requests:**
    *   Listening for and handling server-to-client requests, most notably `sampling/createMessage` and `elicitation/create`.
    *   Managing the user-in-the-loop workflow for these requests, ensuring the user has final approval.

---

### **4. The MCP Server: The Specialist Worker**

The Server is a specialist. Its world is narrowly focused on providing its advertised capabilities. It should be as decoupled as possible from the client and the user.

**Checklist of Server Responsibilities:**

*   **[ ] Implement the Core Logic:**
    *   Containing the actual code that performs the work of a tool (e.g., the code that calls the GitHub API).
    *   Containing the logic to fetch the data for a resource (e.g., the code that runs `git diff`).
    *   Containing the logic to construct the messages for a prompt.

*   **[ ] Listen for and Respond to MCP Requests:**
    *   Listening on a specific transport (`stdio` or an HTTP port).
    *   Parsing incoming JSON-RPC requests.
    *   Routing requests to the correct handler function based on the `method` name.
    *   Returning well-formed JSON-RPC responses (either `result` or `error`).

*   **[ ] Advertise Capabilities:**
    *   Responding to the `prompts/list`, `resources/list`, and `tools/list` methods with accurate, well-described, and schema-complete definitions of all its primitives.
    *   This metadata is the server's public API contract.

*   **[ ] Respect the Security Sandbox (`rootUris`):**
    *   Storing the `rootUris` received during the `initialize` handshake.
    *   **Crucially:** Checking every request against these roots. If a request tries to access a file or resource outside of the allowed scope, the server **MUST** reject it with an error.

*   **[ ] Handle its Own State:**
    *   Managing its own internal state, such as database connections or caches.
    *   The client should not be responsible for managing the server's internal state.

*   **[ ] Send Notifications (Proactively):**
    *   Monitoring its environment for changes.
    *   Sending asynchronous `resources/updated` notifications to the client when a resource has changed (e.g., a file has been modified on disk).
    *   Sending progress notifications for long-running tools.

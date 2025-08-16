
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 1: Foundations of MCP

### Lesson 1.3: The Language of MCP: JSON-RPC 2.0 (13:00 - 14:30)

---

### **1. Why JSON-RPC? The Foundations of a Protocol**

Every great communication protocol is built on a solid foundation. For MCP, that foundation is **JSON-RPC 2.0**. The choice of JSON-RPC was deliberate and strategic, favoring simplicity, ubiquity, and a lightweight footprint over more complex alternatives like gRPC or SOAP.

**What is JSON-RPC 2.0?**

At its core, JSON-RPC 2.0 is a **stateless, light-weight remote procedure call (RPC) protocol**. Let's break that down:

*   **Remote Procedure Call (RPC):** This is a programming paradigm that allows a program on one computer to execute a procedure (a subroutine or function) on another computer without the programmer needing to explicitly code the details for the remote interaction. From the developer's perspective, it looks like they are just calling a local function.
*   **JSON (JavaScript Object Notation):** The data format used to structure the messages. JSON was chosen for its human-readability, its widespread support across virtually all programming languages, and its native compatibility with web technologies.
*   **Stateless:** Each request-response pair is self-contained. The server does not need to remember any previous requests to fulfill the current one. While MCP sessions can have state, the underlying JSON-RPC transport is fundamentally stateless, which is a crucial feature for scalability, especially in HTTP environments.

**The Advantages of Using JSON-RPC for MCP:**

1.  **Simplicity and Readability:** The protocol defines just a handful of data structures (Request, Response, Notification). The messages are plain text and easy to debug. You can literally read the communication between a client and server, which is invaluable during development.
2.  **Transport Agnostic:** JSON-RPC defines the *messages*, not how they are sent. You can send JSON-RPC messages over any transport mechanism you can imagine: HTTP, TCP, WebSockets, or, as is common in MCP, standard I/O (`stdio`). This flexibility is key to MCP's adaptability.
3.  **Widespread Support:** There are mature and robust JSON-RPC 2.0 libraries available for nearly every programming language, from Go and Python to Rust, Java, and TypeScript. This lowers the barrier to entry for developers wanting to build MCP-compliant tools.
4.  **Explicit Error Handling:** The specification has a well-defined structure for error responses, including a set of standard error codes. This encourages robust error handling and makes it easier to build resilient applications.

---

### **2. The Anatomy of a Message: The Request Object**

The Request Object is the workhorse of JSON-RPC. It is the message sent from the Client to the Server to invoke a method.

**Every Request Object MUST contain three fields:** `jsonrpc`, `method`, and `id` (with one exception for Notifications, which we'll cover later).

**The Structure:**

```json
{
  "jsonrpc": "2.0",
  "method": "some_method_name",
  "params": { /* structured data */ },
  "id": 1
}
```

**Field-by-Field Breakdown:**

*   `"jsonrpc"` (string, required):
    *   **Purpose:** Specifies the version of the JSON-RPC protocol.
    *   **Value:** MUST be the string `"2.0"`.
    *   **Why:** This ensures that both the client and server agree on the protocol being used, preventing version mismatch errors.

*   `"method"` (string, required):
    *   **Purpose:** The name of the method to be invoked on the server.
    *   **Value:** A string containing the method name. Method names that begin with `rpc.` are reserved for internal use by the protocol itself and must not be used for application-defined methods.
    *   **MCP Convention:** In MCP, method names are namespaced to avoid collisions and provide clarity. For example:
        *   `tools/list`: List available tools.
        *   `tools/call`: Execute a tool.
        *   `resources/get`: Fetch a resource.
        *   `initialize`: The initial handshake message.

*   `"params"` (object or array, optional):
    *   **Purpose:** A structured value that holds the parameter values to be used during the invocation of the method.
    *   **Value:** This field can be one of two types:
        1.  **By-Position (Array):** The `params` field is an array of values. The server must know the expected order of the parameters.
            ```json
            "params": ["first_param", 42, true]
            ```
        2.  **By-Name (Object):** The `params` field is an object where keys correspond to parameter names. This is the **strongly recommended** and most common approach in MCP, as it is self-documenting and not sensitive to the order of parameters.
            ```json
            "params": {
              "name": "my-project",
              "version": "1.2.3"
            }
            ```
    *   **If omitted:** If the method requires no parameters, the `params` field may be omitted entirely.

*   `"id"` (string, number, or null, required):
    *   **Purpose:** A unique identifier established by the Client. The Server **MUST** reply with a Response Object containing the same `id`.
    *   **Value:** The `id` can be a string, a number, or `null`. It should not be a fractional number. While `null` is technically allowed, it is discouraged because it can be ambiguous.
    *   **Why:** The `id` is the mechanism that allows the Client to match incoming Response messages to the Request messages it previously sent. This is absolutely critical for asynchronous communication, where a client might send multiple requests before receiving any responses. The `id` ensures that the results are correctly correlated.

**Example: A Complete MCP Request Object**

This is a request to call the `github/create_issue` tool.

```json
{
  "jsonrpc": "2.0",
  "id": "request-007",
  "method": "tools/call",
  "params": {
    "name": "github/create_issue",
    "arguments": {
      "repository": "my-org/my-repo",
      "title": "Bug: User cannot log in",
      "body": "When a user with a valid password tries to log in, they receive a 500 error."
    }
  }
}
```

---

### **3. The Server's Reply: The Response Object**

When a server receives a valid Request object, it processes it and replies with a Response Object. The Response Object will indicate either success or failure.

**Every Response Object MUST contain three fields:** `jsonrpc`, `id`, and either `result` or `error`.

**The Structure (Success):**

```json
{
  "jsonrpc": "2.0",
  "result": { /* structured data */ },
  "id": 1
}
```

**The Structure (Error):**

```json
{
  "jsonrpc": "2.0",
  "error": { /* error object */ },
  "id": 1
}
```

**Field-by-Field Breakdown:**

*   `"jsonrpc"` (string, required): Same as the request, must be `"2.0"`.
*   `"id"` (string, number, or null, required): This **MUST** be the same value as the `id` from the Request Object it is responding to.

*   `"result"` (any, required on success):
    *   **Purpose:** Contains the value returned by the server method upon successful invocation.
    *   **Value:** The value can be any valid JSON type: an object, array, string, number, boolean, or `null`. The structure of the `result` is defined by the method that was called.
    *   **Constraint:** This field **MUST NOT** exist if there was an error invoking the method.

*   `"error"` (object, required on failure):
    *   **Purpose:** An object that provides detailed information about the error that occurred.
    *   **Constraint:** This field **MUST NOT** exist if the request was successful.

#### **The Error Object**

The `error` object itself has a defined structure:

```json
{
  "code": -32601,
  "message": "Method not found",
  "data": { /* optional, additional info */ }
}
```

*   `"code"` (integer, required):
    *   **Purpose:** A number that indicates the error type that occurred.
    *   **Value:** JSON-RPC pre-defines a range of error codes. Values from -32768 to -32000 are reserved.

*   `"message"` (string, required):
    *   **Purpose:** A short, single-sentence description of the error.

*   `"data"` (any, optional):
    *   **Purpose:** A primitive or structured value that contains additional, application-specific error information. This is where you can put stack traces, validation error details, etc.

**Standard JSON-RPC Error Codes:**

| Code      | Message            | Meaning                                                                 |
|-----------|--------------------|-------------------------------------------------------------------------|
| -32700    | Parse error        | Invalid JSON was received by the server. An error occurred on the server while parsing the JSON text. |
| -32600    | Invalid Request    | The JSON sent is not a valid Request object.                            |
| -32601    | Method not found   | The method does not exist / is not available.                           |
| -32602    | Invalid Params     | Invalid method parameter(s).                                            |
| -32603    | Internal error     | Internal JSON-RPC error.                                                |
| -32000 to -32099 | Server error | Reserved for implementation-defined server-errors. MCP servers can use this range for their own custom errors. |

**Example: A Success Response**

```json
{
  "jsonrpc": "2.0",
  "id": "request-007",
  "result": {
    "issue_number": 123,
    "url": "https://github.com/my-org/my-repo/issues/123"
  }
}
```

**Example: An Error Response**

```json
{
  "jsonrpc": "2.0",
  "id": "request-008",
  "error": {
    "code": -32602,
    "message": "Invalid Params",
    "data": "The 'repository' field is required and cannot be empty."
  }
}
```

---

### **4. One-Way Communication: The Notification Object**

A Notification is a special type of Request Object. Its purpose is to send a one-way message from the server to the client. The client **SHOULD NOT** reply to a Notification.

**How is it different from a Request?**

A Notification is a Request object that is missing the `id` field.

**The Structure:**

```json
{
  "jsonrpc": "2.0",
  "method": "some_notification_name",
  "params": { /* structured data */ }
}
```

By omitting the `id`, the server is signaling that it does not expect a response. This is a "fire-and-forget" mechanism.

**Use Cases in MCP:**

Notifications are crucial for allowing the server to proactively send information to the client without being asked.

*   **Resource Updates:** This is the most common use case. A server that is monitoring a file system can send a `resources/updated` notification to the client when a file is changed on disk. This allows the client (e.g., an IDE) to know that its view of the file is stale and needs to be refreshed.
*   **Progress Updates:** For a long-running tool, the server could send periodic `tool/progress` notifications to the client to update a progress bar in the UI.
*   **Server-Side Events:** A server might send a notification to indicate that a background task has completed or that an external event has occurred.

**Example: A File Update Notification**

The `git-mcp-server` might be watching the git index. When the user stages a new file in a separate terminal, the server detects this change and sends the following notification to the client.

```json
{
  "jsonrpc": "2.0",
  "method": "resources/updated",
  "params": {
    "uri": "git:/diff/staged"
  }
}
```

Upon receiving this, the client knows that the staged diff has changed and can re-fetch the resource if it needs the latest version.

This deep understanding of the three message types—Request, Response, and Notification—is the key to unlocking the full power and predictability of the Model Context Protocol. They are the simple, robust building blocks for all communication.

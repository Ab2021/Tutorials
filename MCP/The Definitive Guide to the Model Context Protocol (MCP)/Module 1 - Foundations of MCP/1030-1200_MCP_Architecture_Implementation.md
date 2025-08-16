
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 1: Foundations of MCP

### Lesson 1.2: The MCP Architecture (10:30 - 12:00)

### **Implementation & Examples**

---

### **1. Implementing the Components: A Practical Blueprint**

Let's translate the architectural theory into more concrete implementation details. We'll use a practical scenario: **An AI-powered IDE (the Host) that connects to a local `linter` MCP Server to check code for errors.**

This example will showcase:
*   How a Host launches and connects to a local server.
*   The clear separation of responsibilities.
*   A tangible view of the JSON-RPC messages being passed back and forth.

#### **Component 1: The Host & MCP Client Implementation (Conceptual Go Code)**

The Host is the IDE. The MCP Client is a library or a module within the IDE's codebase. Here's a conceptual look at how the client-side logic would be structured in Go.

```go
// package ide_main

import (
    "fmt"
    "os/exec"
    "path/to/mcp/client"
)

// The Host application's main structure.
type IDEApplication struct {
    mcpClient *client.MCPClient
    ui        *UserInterface
}

func NewIDEApplication() *IDEApplication {
    ide := &IDEApplication{
        ui: NewUserInterface(),
    }

    // The Host is responsible for configuring and launching the server.
    // This is a critical piece of the setup.
    serverPath := "./bin/linter-mcp-server" // Path to the server executable
    cmd := exec.Command(serverPath)

    // The client library handles the low-level process management and communication.
    mcpClient, err := client.NewStdioMCPClient(cmd)
    if err != nil {
        log.Fatalf("Failed to start MCP server: %v", err)
    }

    ide.mcpClient = mcpClient
    return ide
}

// This function is called when the user clicks a "Lint File" button in the IDE.
func (ide *IDEApplication) onLintFileButtonClick(filePath string) {
    // The Host UI triggers the action.
    // It knows the file path from its own state.

    // 1. The Host tells the Client to call a specific tool.
    // It doesn't know how the tool works, just its name and expected parameters.
    toolName := "linter/run"
    params := map[string]interface{}{
        "file_uri": "file://" + filePath, // Constructing the resource URI
    }

    // The client library abstracts away the JSON-RPC complexity.
    result, err := ide.mcpClient.CallTool(toolName, params)
    if err != nil {
        ide.ui.ShowError(fmt.Sprintf("Linter tool failed: %v", err))
        return
    }

    // 4. The Host receives the structured result and updates the UI.
    // The result is just data. The Host decides how to present it.
    lintIssues := result.Get("issues").([]LintIssue) // Assuming type assertion
    ide.ui.DisplayLintIssues(filePath, lintIssues)
}

func (ide *IDEApplication) Shutdown() {
    // When the IDE closes, it gracefully shuts down the client and the server.
    ide.mcpClient.Shutdown()
}

```

**Key Takeaways from the Client-Side Implementation:**

*   **Ownership:** The Host *owns* the server process. It decides when to start it (`NewIDEApplication`) and when to stop it (`Shutdown`).
*   **Abstraction:** The Host interacts with a high-level `MCPClient` object. It doesn't manually create JSON objects or manage `stdin`/`stdout` pipes. It just calls a method like `CallTool`.
*   **Data, Not Logic:** The Host sends and receives simple data structures (maps, strings, etc.). The *business logic* of linting resides entirely on the server.

--- 

#### **Component 2: The MCP Server Implementation (Conceptual Go Code)**

Now let's look at the other side of the connection: the `linter-mcp-server`. This is a completely separate Go program.

```go
// package main (linter-mcp-server)

import (
    "path/to/mcp/server"
    "path/to/linter/engine"
)

// The handler function for our tool.
// This is the core logic of our server.
func handleLinterRun(req *mcp.Request) (*mcp.Result, *mcp.Error) {
    // 2. The Server receives the request and parses the parameters.
    // The server library would provide helpers for this.
    filePath, ok := req.Params["file_uri"].(string)
    if !ok {
        return nil, mcp.NewInvalidParamsError("file_uri is required and must be a string")
    }

    // The server is responsible for its own business logic.
    // Here, it calls a hypothetical linting engine.
    linterEngine := linter.NewEngine()
    issues, err := linterEngine.Run(filePath) // The actual work gets done here.
    if err != nil {
        // If the linter itself fails, return a custom error.
        return nil, mcp.NewError(5000, "Linter engine failed", err)
    }

    // 3. The Server constructs a successful result with structured data.
    resultData := map[string]interface{}{
        "issues": issues, // `issues` would be a slice of structs
    }

    return mcp.NewResult(resultData), nil
}

func main() {
    // The main function of the server is to define its capabilities and start listening.
    s := server.NewMCPServer("GoLinterServer", "1.0.0")

    // Define the tool's schema. This is critical for discoverability.
    linterTool := mcp.NewTool(
        "linter/run",
        "Runs the linter on a given file URI and returns a list of issues.",
    ).WithInputSchema(mcp.Object(
        mcp.Property("file_uri", mcp.String().WithDescription("The URI of the file to lint."), mcp.Required()),
    ))

    // Register the tool and its handler function.
    s.AddTool(linterTool, handleLinterRun)

    // Start the server, listening on standard I/O.
    // This call will block until the client closes the connection.
    err := server.ServeStdio(s)
    if err != nil {
        log.Fatalf("Server exited with error: %v", err)
    }
}
```

**Key Takeaways from the Server-Side Implementation:**

*   **Self-Contained:** The server is a complete, runnable program. It has no dependency on the IDE/Host.
*   **Contract-Driven:** The server explicitly defines the `linter/run` tool and its input schema. This is its public API contract.
*   **Focused Logic:** The server's only job is to lint files. It doesn't know about UI, LLMs, or anything else. This makes it easy to test, maintain, and reuse.

--- 

### **2. Visualizing the Communication: The JSON-RPC Messages**

Let's trace the `onLintFileButtonClick` call and look at the actual JSON-RPC 2.0 messages that would be passed over the `stdio` pipe between the Client and Server.

**Step 1: The Client sends a `tools/call` Request**

When `ide.mcpClient.CallTool(...)` is executed, the client library constructs and sends the following JSON object to the server's standard input. Note that we add a newline character to signal the end of the message.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "linter/run",
    "arguments": {
      "file_uri": "file:///path/to/my/project/main.go"
    }
  }
}

```

*   `"jsonrpc": "2.0"`: Specifies the protocol version.
*   `"id": 1`: A unique identifier for this request. The response *must* have the same ID.
*   `"method": "tools/call"`: The standard MCP method for executing a tool.
*   `"params"`: An object containing the parameters for the `tools/call` method itself, which includes the `name` of the tool to run and the `arguments` for that specific tool.

**Step 2: The Server sends a `tools/call` Response**

The server receives the message, routes it to the `handleLinterRun` function, executes the linter, and gets a list of issues. It then constructs and sends the following JSON response back to the client's standard output.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "issues": [
      {
        "line": 15,
        "column": 8,
        "severity": "error",
        "message": "Undeclared name: `logf`"
      },
      {
        "line": 22,
        "column": 5,
        "severity": "warning",
        "message": "Variable `err` is unused"
      }
    ]
  }
}

```

*   `"id": 1`: The ID matches the request, so the client knows which request this response corresponds to.
*   `"result"`: This key indicates a successful execution. Its value contains the structured data returned by the `handleLinterRun` function.

**Alternative Step 2: The Server sends an Error Response**

What if the `file_uri` parameter was missing? The server's validation logic would catch this, and it would return a JSON-RPC Error Object instead.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32602,
    "message": "Invalid params",
    "data": "file_uri is required and must be a string"
  }
}

```

*   `"error"`: This key indicates that something went wrong.
*   `"code": -32602`: The standard JSON-RPC code for invalid parameters.
*   `"message"`: A short description of the error.
*   `"data"`: Optional, can contain more specific information about the error.

This explicit, structured communication is the core of MCP. It's predictable, machine-readable, and provides a clear contract between the Client and the Server, completely independent of the programming languages they are written in. This example demonstrates how the architectural components translate directly into robust, decoupled code.

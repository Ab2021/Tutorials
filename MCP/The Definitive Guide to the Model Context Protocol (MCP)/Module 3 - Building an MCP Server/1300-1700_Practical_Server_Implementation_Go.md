# The Definitive Guide to the Model Context Protocol (MCP)

## Module 3: Building an MCP Server: The Three Primitives

### Lesson 3.3: Practical Server Implementation (Go) (13:00 - 17:00)

---

### **1. Introduction: From Theory to Production Code**

In the previous lessons, we have developed a strong conceptual understanding of the three server primitives. We have seen how they are defined and how they fit into the control level hierarchy. Now, it is time to translate that theory into practice. This lesson will guide you through the process of building a complete, functional MCP server from scratch using the official **`mcp-go`** library.

**Why Go?**

Go is an excellent language for building MCP servers for several reasons:
*   **Performance:** Go is a compiled language that is known for its speed and efficiency, making it well-suited for building high-performance servers.
*   **Concurrency:** Go has first-class support for concurrency with goroutines and channels, which makes it easy to handle multiple requests and long-running tasks without blocking.
*   **Static Typing:** Go's static type system helps to catch errors at compile time, leading to more robust and reliable code.
*   **Strong Standard Library:** Go's standard library provides excellent support for networking, I/O, and JSON manipulation, which are all essential for building MCP servers.

**The `mcp-go` Library**

The `mcp-go` library is the official Go implementation of the Model Context Protocol. It provides a set of high-level APIs that abstract away the complexities of the JSON-RPC 2.0 protocol, allowing you to focus on implementing your server's capabilities. It includes helpers for:

*   Creating server instances.
*   Defining and registering Prompts, Resources, and Tools with fluent, chainable APIs.
*   Handling requests and sending responses.
*   Serving over different transports like `stdio` and HTTP.

**Our Project: The "DevBox" Server**

In this lesson, we will build a simple but practical "DevBox" server. This server will provide a few common capabilities that a developer might find useful:

1.  **A Resource:** It will expose the current working directory's file listing as a resource.
2.  **A Tool:** It will provide a tool to read the contents of a specific file.
3.  **A Prompt:** It will offer a prompt to help generate a `.gitignore` file for a project.

By the end of this lesson, you will have a runnable MCP server that you can interact with using a separate client.

---

### **2. Setup: Getting Your Go Environment Ready**

Before we start coding, make sure you have a working Go environment.

1.  **Install Go:** If you don't have Go installed, download it from the [official Go website](https://golang.org/dl/).
2.  **Create a Project Directory:**
    ```bash
    mkdir devbox-mcp-server
    cd devbox-mcp-server
    ```
3.  **Initialize a Go Module:**
    ```bash
    go mod init github.com/your-username/devbox-mcp-server
    ```
4.  **Install the `mcp-go` Library:**
    ```bash
    go get github.com/sourcegraph/mcp-go/server
    ```
    *(Note: The exact import path may vary, always check the official library documentation.)*

Now, create a `main.go` file in your project directory. This is where we will write our server code.

---

### **3. Step 1: The Server Skeleton**

Every `mcp-go` server starts with the same basic structure: creating a server instance. This instance acts as the registry for all the capabilities you will add.

```go
// main.go
package main

import (
	"context"
	"log"

	"github.com/sourcegraph/mcp-go/server"
)

func main() {
	// Create a new MCP server instance with a name and version.
	// These are important for the `initialize` handshake.
	s := server.NewMCPServer("DevBoxServer", "1.0.0")

	// We will add our primitives here in the next steps.

	// Create a context to manage the server's lifecycle.
	ctx := context.Background()

	// Start the server, listening for messages on standard I/O.
	// The ServeStdio function will block until the client closes the connection.
	log.Println("DevBox MCP Server starting on stdio...")
	if err := server.ServeStdio(ctx, s); err != nil {
		log.Fatalf("Server exited with error: %v", err)
	}
	log.Println("Server shutting down.")
}
```

**Code Breakdown:**

*   `server.NewMCPServer("DevBoxServer", "1.0.0")`: This is the first and most important call. It creates the server object, `s`, and sets its name and version. This information is used by the client during the initial handshake to identify the server it's talking to.
*   `context.Background()`: We create a background context to control the server's lifecycle. This is a standard pattern in modern Go applications.
*   `server.ServeStdio(ctx, s)`: This is the blocking call that starts the server. It tells the server to listen for JSON-RPC messages on its standard input (`stdin`) and send responses on its standard output (`stdout`). The program will sit here, waiting for and processing messages, until the client closes the connection or the context is canceled.

At this point, you could already run `go run main.go`. It would start, print the log message, and wait. It doesn't have any capabilities yet, so let's add them.

---

### **4. Step 2: Implementing a Resource**

Next, let's add a tool that can read the content of a specific file.

**The Definition:** A tool named `file/read` that takes one argument, `path` (a string), and returns the file's content.

**The Implementation:**

First, the handler.

```go
readFileHandler := func(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("Handling request for file/read tool...")

	// The arguments for a tool call are in a nested `arguments` field.
	// The mcp-go library could provide helpers for this, but we do it manually here for clarity.
	params, ok := req.Params.(map[string]any)
	if !ok { return nil, &protocol.Error{Code: -32602, Message: "Invalid params"} }
	args, ok := params["arguments"].(map[string]any)
	if !ok { return nil, &protocol.Error{Code: -32602, Message: "Invalid params"} }

	path, ok := args["path"].(string)
	if !ok {
		return nil, &protocol.Error{Code: -32602, Message: "Invalid params", Data: "'path' argument is required and must be a string."} 
	}

	content, err := ioutil.ReadFile(path)
	if err != nil {
		return nil, &protocol.Error{Code: -32000, Message: "Server error", Data: "Failed to read file: " + err.Error()}
	}

	return string(content), nil
}
```

Now, register it in `main()`.

```go
// Inside main()

// Define the tool and its input schema.
readFileTool := protocol.NewTool(
	"file/read",
	"Reads the content of a specific file."
).WithInputSchema(protocol.Object(
	protocol.Property("path", protocol.String().WithDescription("The path to the file to read."), protocol.Required()),
))

// Register the tool and its handler.
s.AddTool(readFileTool, server.NewToolHandler(readFileHandler))
```

**Code Breakdown:**

*   **Argument Parsing:** Tool handlers are more complex than resource handlers because they need to parse arguments. We have to safely cast the `req.Params` to access the nested `arguments` map and then extract the `path`.
*   **Error Handling:** We return a specific `-32602 Invalid Params` error if the `path` argument is missing or not a string. We return a custom `-32000` server error if `ioutil.ReadFile` fails.
*   `protocol.NewTool(...)`: The helper for defining a tool.
*   `WithInputSchema(protocol.Object(...))`: This is where the `mcp-go` library shines. It provides a set of helpers (`Object`, `Property`, `String`, `Required`) to build the JSON Schema for the tool's input in a programmatic and type-safe way. This is much less error-prone than writing raw JSON strings.

---

### **6. Step 4: Implementing a Prompt**

Finally, let's add our `.gitignore` prompt.

**The Definition:** A prompt named `gitignore/generate` that takes one argument, `project_type` (e.g., "go", "node", "python"), and generates a recommended `.gitignore` file.

**The Implementation:**

First, the handler.

```go
// gitignorePromptHandler is the handler for our prompt.
func gitignorePromptHandler(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("Handling request for gitignore/generate prompt...")

	// Prompts also have arguments.
	args, ok := req.Params.(map[string]any)
	if !ok { return nil, &protocol.Error{Code: -32602, Message: "Invalid params"} }

	projectType, ok := args["project_type"].(string)
	if !ok {
		return nil, &protocol.Error{Code: -32602, Message: "Invalid params", Data: "'project_type' argument is required."} 
	}

	// This is a simplified logic. A real server might fetch templates from a remote URL.
	var gitignoreContent string
	switch projectType {
	case "go":
		gitignoreContent = "# Go\n*.exe\n*.out\n/vendor/\n"
	case "node":
		gitignoreContent = "# Node\nnode_modules/\nnpm-debug.log\n"
	default:
		gitignoreContent = "# Generic\n.DS_Store\n"
	}

	// The result of a prompt handler is a `GetPromptResult` object.
	// We construct it with a system message and a user message.
	promptResult := &protocol.GetPromptResult{
		Messages: []protocol.PromptMessage{
			{Speaker: "system", Text: "You are a helpful assistant that generates .gitignore files."},
			{Speaker: "user", Text: "Generate a .gitignore file for a '" + projectType + "' project. Here is a recommended template:\n\n" + gitignoreContent},
		},
	}

	return promptResult, nil
}
```

And register it in `main()`.

```go
// Inside main()

// Define the prompt and its arguments.
gitignorePrompt := protocol.NewPrompt(
	"gitignore/generate",
	"Generates a .gitignore file for a given project type."
).WithArgument("project_type", protocol.String(), "The type of project (e.g., 'go', 'node').", protocol.Required())

// Register the prompt and its handler.
s.AddPrompt(gitignorePrompt, server.NewPromptHandler(gitignorePromptHandler))
```

**Code Breakdown:**

*   `protocol.GetPromptResult`: The key difference in a prompt handler is the return type. It doesn't return raw data; it returns a `*protocol.GetPromptResult` struct. This struct contains the list of `PromptMessage` objects that the client should send to the LLM.
*   `protocol.PromptMessage`: Each message has a `Speaker` (`system` or `user`) and the `Text` of the message.
*   `protocol.NewPrompt(...)`: The helper for defining a prompt.
*   `.WithArgument(...)`: A convenient helper for the common case of adding a simple argument to a prompt.

Your `main.go` file is now complete! You have a fully functional MCP server with three different kinds of capabilities. The next step is to run it and interact with it using a client.

```
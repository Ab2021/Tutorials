# The Definitive Guide to the Model Context Protocol (MCP)

## Module 4: Building an MCP Client: Interacting with Servers

### Lesson 4.2: Practical Client Implementation (Go) (10:30 - 12:00)

---

### **1. Introduction: Building the Orchestrator**

In the last module, we built our `DevBoxServer`, a fully functional MCP server with a resource, a tool, and a prompt. However, a server is useless without a client to talk to it. This lesson will complete the picture by guiding you through the creation of a command-line client in Go that will connect to and interact with our `DevBoxServer`.

This client will be the orchestrator. It will be responsible for:

*   Launching the server process.
*   Connecting to it over `stdio`.
*   Performing the initial `initialize` handshake.
*   Discovering the server's capabilities.
*   Executing each of the three primitives we implemented.

We will use the official **`mcp-go/client`** library, which provides a high-level interface for managing the client side of an MCP connection, complementing the `server` library we used previously.

By the end of this lesson, you will have a complete, end-to-end MCP application, demonstrating the full communication lifecycle between a client and a server.

---

### **2. Setup: Preparing the Client Project**

We will create a new Go project for our client. It's important to keep the client and server code separate to emphasize their decoupled nature.

1.  **Create a Project Directory:** From the parent directory of your `devbox-mcp-server` project, create a new directory for the client.
    ```bash
    mkdir devbox-mcp-client
    cd devbox-mcp-client
    ```
2.  **Initialize a Go Module:**
    ```bash
    go mod init github.com/your-username/devbox-mcp-client
    ```
3.  **Install the `mcp-go` Client Library:**
    ```bash
    go get github.com/sourcegraph/mcp-go/client
    ```
4.  **Create a `main.go` file:** This is where we will write our client code.

---

### **3. Step 1: Creating and Connecting the Client**

The first step is to write the boilerplate code to launch the server process and establish a connection. The `mcp-go/client` library makes this remarkably simple.

```go
// main.go (in devbox-mcp-client)
package main

import (
	"context"
	"log"
	"os/exec"

	"github.com/sourcegraph/mcp-go/client"
	"github.com/sourcegraph/mcp-go/protocol"
)

func main() {
	ctx := context.Background()

	// Define the command to run the server.
	// This assumes the server executable is in the specified path.
	// You might need to adjust this path based on your setup.
	cmd := exec.CommandContext(ctx, "go", "run", "../devbox-mcp-server/main.go")

	// Create a new stdio client. This function handles launching the process
	// and setting up the stdin/stdout pipes.
	mcpClient, err := client.NewStdioMCPClient(ctx, cmd)
	if err != nil {
		log.Fatalf("Failed to create MCP client: %v", err)
	}
	// Ensure the client and its underlying server process are cleaned up.
	defer mcpClient.Shutdown(ctx)

	log.Println("Client created and server process launched.")

	// In the next steps, we will add the code to interact with the server here.
}
```

**Code Breakdown:**

*   `exec.CommandContext(...)`: We create a command object to run our server. Using `go run` is convenient for development. In production, you would compile your server to a binary and run that directly.
*   `client.NewStdioMCPClient(ctx, cmd)`: This is the core function from the client library. It takes the command, launches it, and returns a `*client.MCPClient` object that is ready to use. It handles all the low-level process and pipe management.
*   `defer mcpClient.Shutdown(ctx)`: This is a crucial line. The `defer` keyword ensures that `mcpClient.Shutdown()` is called when the `main` function exits. This function gracefully terminates the server process and closes the connection, preventing orphaned processes.

---

### **4. Step 2: The Handshake (Initialize)**

Before we can do anything else, we must perform the `initialize` handshake. This is the first message sent on any new MCP connection. It tells the server who we are and what version of the protocol we support.

Add this code inside your `main` function after the client is created.

```go
// Inside main()

// The Initialize request is the first message sent.
	initParams := protocol.InitializeParams{
		ClientName:    "DevBoxClient",
		ClientVersion: "1.0.0",
		ProtocolVersion: "1.0", // Specify the MCP version you support
	}

	serverInfo, err := mcpClient.Initialize(ctx, initParams)
	if err != nil {
		log.Fatalf("MCP Handshake failed: %v", err)
	}

	log.Printf("Successfully connected to server: %s %s", serverInfo.ServerName, serverInfo.ServerVersion)
```

**Code Breakdown:**

*   `protocol.InitializeParams{...}`: We create a struct that holds the information for the `initialize` request.
*   `mcpClient.Initialize(ctx, initParams)`: This is a high-level helper method on the client object. It sends the `initialize` request and waits for the server's response. It handles the JSON-RPC serialization and deserialization for you.
*   `serverInfo`: On success, the `Initialize` method returns a struct containing the server's name and version, which it declared when it was created. This is a great way to confirm you've connected to the correct server.

---

### **5. Step 3: Discovery and Execution**

Now that we are connected, we can interact with the server's primitives. We will discover the available capabilities and then execute each one.

#### **Listing and Getting a Resource**

Let's get the list of files in the server's directory.

```go
// Inside main(), after the handshake

// Discover available resources
	log.Println("--- Discovering Resources ---")
	resources, err := mcpClient.ListResources(ctx)
	if err != nil {
		log.Fatalf("Failed to list resources: %v", err)
	}
	for _, r := range resources {
		log.Printf("Found Resource: %s (%s)\n", r.Name, r.URI)
	}

	// Get the content of our specific resource
	log.Println("--- Getting Resource Content ---")
	var fileList []string
	err = mcpClient.GetResource(ctx, "file:///./", &fileList)
	if err != nil {
		log.Fatalf("Failed to get resource: %v", err)
	}
	log.Printf("Files in server directory: %v\n", fileList)
```

**Code Breakdown:**

*   `mcpClient.ListResources(ctx)`: A helper that calls the `resources/list` method and returns a slice of `protocol.Resource` structs.
*   `mcpClient.GetResource(ctx, "file:///./", &fileList)`: This is a generic helper for calling `resources/get`. You provide the URI of the resource and a pointer to a variable where the result should be deserialized. The library uses reflection to handle the JSON unmarshaling, so `fileList` will be populated with the array of strings from the server's response.

#### **Listing and Calling a Tool**

Now, let's use our `file/read` tool to read the content of the server's `main.go` file.

```go
// Inside main()

// Discover available tools
	log.Println("--- Discovering Tools ---")
	tools, err := mcpClient.ListTools(ctx)
	if err != nil {
		log.Fatalf("Failed to list tools: %v", err)
	}
	for _, t := range tools {
		log.Printf("Found Tool: %s\n", t.Name)
	}

	// Call the file/read tool
	log.Println("--- Calling Tool ---")
	toolParams := map[string]any{
		"path": "../devbox-mcp-server/main.go", // The path is relative to the server's CWD
	}
	var fileContent string
	err = mcpClient.CallTool(ctx, "file/read", toolParams, &fileContent)
	if err != nil {
		log.Fatalf("Failed to call tool: %v", err)
	}
	log.Printf("Content of server's main.go:\n%s\n", fileContent)
```

**Code Breakdown:**

*   `mcpClient.ListTools(ctx)`: Similar to listing resources, this discovers the available tools.
*   `mcpClient.CallTool(ctx, "file/read", toolParams, &fileContent)`: The primary method for executing a tool. You provide the tool name, a map of arguments, and a pointer to a variable for the result.

#### **Listing and Getting a Prompt**

Finally, let's get our `.gitignore` prompt. Note that we are not interacting with an LLM in this client; we are just demonstrating how the client fetches the prompt definition from the server.

```go
// Inside main()

// Discover available prompts
	log.Println("--- Discovering Prompts ---")
	prompts, err := mcpClient.ListPrompts(ctx)
	if err != nil {
		log.Fatalf("Failed to list prompts: %v", err)
	}
	for _, p := range prompts {
		log.Printf("Found Prompt: %s\n", p.Name)
	}

	// Get the prompt from the server
	log.Println("--- Getting Prompt ---")
	promptArgs := map[string]any{
		"project_type": "go",
	}
	promptResult, err := mcpClient.GetPrompt(ctx, "gitignore/generate", promptArgs)
	if err != nil {
		log.Fatalf("Failed to get prompt: %v", err)
	}
	log.Println("Received prompt messages from server:")
	for _, msg := range promptResult.Messages {
		log.Printf("  Speaker: %s, Text: %s\n", msg.Speaker, msg.Text)
	}
```

**Code Breakdown:**

*   `mcpClient.ListPrompts(ctx)`: Discovers available prompts.
*   `mcpClient.GetPrompt(ctx, "gitignore/generate", promptArgs)`: This calls the `prompts/get` method. It takes the prompt name and a map of arguments. It returns a `*protocol.GetPromptResult` struct, which contains the list of messages that would be sent to the LLM.

This completes our client. When you run this code, it will connect to the server, perform the handshake, and then systematically discover and execute every capability the server offers. You have now built a full end-to-end MCP application, demonstrating the complete lifecycle of communication and interaction.

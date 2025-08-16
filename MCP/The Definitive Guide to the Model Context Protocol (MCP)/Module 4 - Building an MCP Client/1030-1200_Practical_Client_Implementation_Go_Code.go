// This file contains the complete, runnable code for the DevBox MCP client.
// To run this code:
// 1. Make sure you have the `devbox-mcp-server` project from Module 3 in a sibling directory.
// 2. Create a directory for this client, save this file as `main.go`.
// 3. Open a terminal in that directory.
// 4. Run `go mod init devbox-client`
// 5. Run `go get github.com/sourcegraph/mcp-go/client` (or the correct path for the library)
// 6. Run `go run main.go`

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

	// --- Step 1: Create and Connect the Client ---

	// Define the command to run the server.
	// This uses `go run` to execute the server's main.go file from the sibling directory.
	// Adjust the path if your project structure is different.
	cmd := exec.CommandContext(ctx, "go", "run", "../devbox-mcp-server/")

	// Create a new stdio client. This function handles launching the process
	// and setting up the stdin/stdout pipes.
	mcpClient, err := client.NewStdioMCPClient(ctx, cmd)
	if err != nil {
		log.Fatalf("Failed to create MCP client: %v", err)
	}
	// Ensure the client and its underlying server process are cleaned up when main exits.
	defer mcpClient.Shutdown(ctx)

	log.Println("Client created and server process launched.")

	// --- Step 2: The Handshake (Initialize) ---

	// The Initialize request is the first message sent on a new connection.
	initParams := protocol.InitializeParams{
		ClientName:      "DevBoxClient",
		ClientVersion:   "1.0.0",
		ProtocolVersion: "1.0",
	}

	serverInfo, err := mcpClient.Initialize(ctx, initParams)
	if err != nil {
		log.Fatalf("MCP Handshake failed: %v", err)
	}

	log.Printf("Successfully connected to server: %s %s\n", serverInfo.ServerName, serverInfo.ServerVersion)

	// --- Step 3: Discovery and Execution ---

	// Discover and get a Resource
	log.Println("--- Discovering Resources ---")
	resources, err := mcpClient.ListResources(ctx)
	if err != nil {
		log.Fatalf("Failed to list resources: %v", err)
	}
	for _, r := range resources {
		log.Printf("Found Resource: %s (%s)", r.Name, r.URI)
	}

	log.Println("--- Getting Resource Content ---")
	var fileList []string
	// We provide the URI of the resource and a pointer to a variable for the result.
	err = mcpClient.GetResource(ctx, "file:///./", &fileList)
	if err != nil {
		log.Fatalf("Failed to get resource: %v", err)
	}
	log.Printf("Files in server directory: %v\n", fileList)

	// Discover and call a Tool
	log.Println("--- Discovering Tools ---")
	tools, err := mcpClient.ListTools(ctx)
	if err != nil {
		log.Fatalf("Failed to list tools: %v", err)
	}
	for _, t := range tools {
		log.Printf("Found Tool: %s", t.Name)
	}

	log.Println("--- Calling Tool ---")
	toolParams := map[string]any{
		// Note: The path is relative to the SERVER's working directory.
		"path": "main.go",
	}
	var fileContent string
	// We provide the tool name, arguments, and a pointer for the result.
	err = mcpClient.CallTool(ctx, "file/read", toolParams, &fileContent)
	if err != nil {
		log.Fatalf("Failed to call tool: %v", err)
	}
	log.Printf("Content of server's main.go (first 100 chars):\n%s...\n", fileContent[:100])

	// Discover and get a Prompt
	log.Println("--- Discovering Prompts ---")
	prompts, err := mcpClient.ListPrompts(ctx)
	if err != nil {
		log.Fatalf("Failed to list prompts: %v", err)
	}
	for _, p := range prompts {
		log.Printf("Found Prompt: %s", p.Name)
	}

	log.Println("--- Getting Prompt ---")
	promptArgs := map[string]any{
		"project_type": "go",
	}
	// We provide the prompt name and arguments.
	promptResult, err := mcpClient.GetPrompt(ctx, "gitignore/generate", promptArgs)
	if err != nil {
		log.Fatalf("Failed to get prompt: %v", err)
	}
	log.Println("Received prompt messages from server:")
	for _, msg := range promptResult.Messages {
		log.Printf("  Speaker: %s, Text: %s", msg.Speaker, msg.Text)
	}

	log.Println("\nClient finished successfully.")
}

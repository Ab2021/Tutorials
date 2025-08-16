// This file contains the complete, runnable code for the DevBox MCP server.
// To run this code:
// 1. Make sure you have Go installed (https://golang.org/dl/).
// 2. Create a directory and save this file as `main.go`.
// 3. Open a terminal in that directory.
// 4. Run `go mod init devbox-server`
// 5. Run `go get github.com/sourcegraph/mcp-go/server` (or the correct path for the library)
// 6. Run `go run main.go`

package main

import (
	"context"
	"io/ioutil"
	"log"

	"github.com/sourcegraph/mcp-go/protocol"
	"github.com/sourcegraph/mcp-go/server"
)

// --- Resource Implementation ---

// listFilesHandler is the function that will be called when a client requests our resource.
func listFilesHandler(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("Handling request for file list resource...")

	files, err := ioutil.ReadDir(".") // Read the current directory
	if err != nil {
		// If something goes wrong, return a standard JSON-RPC internal error.
		return nil, &protocol.Error{Code: -32603, Message: "Internal error", Data: err.Error()}
	}

	var fileNames []string
	for _, file := range files {
		fileNames = append(fileNames, file.Name())
	}

	// The handler returns the data that will be placed in the `result` field.
	// It will be automatically marshaled to JSON.
	return fileNames, nil
}

// --- Tool Implementation ---

// readFileHandler is the handler for our tool.
func readFileHandler(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("Handling request for file/read tool...")

	// The arguments for a tool call are in a nested `arguments` field.
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

// --- Prompt Implementation ---

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

// --- Main Server Function ---

func main() {
	// Create a new MCP server instance with a name and version.
	s := server.NewMCPServer("DevBoxServer", "1.0.0")

	// --- Register Resource ---
	listResource := protocol.NewResource(
		"file:///./",
		"Current Directory Listing",
		"application/json",
	).WithDescription("Lists all files and folders in the server's current working directory.")
	s.AddResource(listResource, server.NewResourceHandler(listFilesHandler))

	// --- Register Tool ---
	readFileTool := protocol.NewTool(
		"file/read",
		"Reads the content of a specific file.",
	).WithInputSchema(protocol.Object(
		protocol.Property("path", protocol.String().WithDescription("The path to the file to read."), protocol.Required()),
	))
	s.AddTool(readFileTool, server.NewToolHandler(readFileHandler))

	// --- Register Prompt ---
	gitignorePrompt := protocol.NewPrompt(
		"gitignore/generate",
		"Generates a .gitignore file for a given project type.",
	).WithArgument("project_type", protocol.String(), "The type of project (e.g., 'go', 'node').", protocol.Required())
	s.AddPrompt(gitignorePrompt, server.NewPromptHandler(gitignorePromptHandler))

	// --- Start Server ---
	ctx := context.Background()
	log.Println("DevBox MCP Server starting on stdio...")
	if err := server.ServeStdio(ctx, s); err != nil {
		log.Fatalf("Server exited with error: %v", err)
	}
	log.Println("Server shutting down.")
}

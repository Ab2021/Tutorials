
// This file contains the complete, runnable code for the IP Geolocation MCP server.
// To run this code:
// 1. Make sure you have Go installed.
// 2. Create a directory and save this file as `main.go`.
// 3. Open a terminal in that directory.
// 4. Run `go mod init ip-server`
// 5. Run `go get github.com/sourcegraph/mcp-go/server`
// 6. Run `go run main.go`

package main

import (
	"context"
	"encoding/json"
	"fmt"
	"log"
	"net/http"

	"github.com/sourcegraph/mcp-go/protocol"
	"github.com/sourcegraph/mcp-go/server"
)

//======================================================================================
// LAYER 1: SERVICE LAYER
// This layer is responsible for interacting with the external ip-api.com API.
// It knows nothing about MCP.
//======================================================================================

const apiBaseURL = "http://ip-api.com/json/"

// IPDetails corresponds to the JSON structure returned by the ip-api.com API.
// We use json tags to map the snake_case JSON fields to our CamelCase Go fields.
type IPDetails struct {
	Status  string  `json:"status"`
	Country string  `json:"country"`
	City    string  `json:"city"`
	Lat     float64 `json:"lat"`
	Lon     float64 `json:"lon"`
	ISP     string  `json:"isp"`
	Query   string  `json:"query"`
	Message string  `json:"message"` // Used for error responses from the API
}

// GetIPDetails fetches geolocation details for a given IP address.
func GetIPDetails(ctx context.Context, ipAddress string) (*IPDetails, error) {
	// Construct the full API URL.
	url := apiBaseURL + ipAddress

	// Create a new HTTP request with the context.
	req, err := http.NewRequestWithContext(ctx, "GET", url, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create request: %w", err)
	}

	// Make the HTTP request.
	log.Printf("SERVICE: Calling external API: %s", url)
	resp, err := http.DefaultClient.Do(req)
	if err != nil {
		return nil, fmt.Errorf("failed to execute request: %w", err)
	}
	defer resp.Body.Close()

	// Decode the JSON response into our struct.
	var details IPDetails
	if err := json.NewDecoder(resp.Body).Decode(&details); err != nil {
		return nil, fmt.Errorf("failed to decode response: %w", err)
	}

	// The API returns a 200 OK even for errors, so we must check the status field.
	if details.Status == "fail" {
		return nil, fmt.Errorf("API error: %s", details.Message)
	}

	return &details, nil
}

//======================================================================================
// LAYER 2: TOOLS LAYER
// This layer is the bridge between the MCP protocol and the service layer.
//======================================================================================

// HandleIPGeolocate is the handler for the `ip/geolocate` tool.
func HandleIPGeolocate(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("HANDLER: Handling request for ip/geolocate tool...")

	// 1. Parse and validate the input argument from the MCP request.
	params, _ := req.Params.(map[string]any)
	args, _ := params["arguments"].(map[string]any)
	ip, ok := args["ip_address"].(string)
	if !ok {
		return nil, &protocol.Error{Code: -32602, Message: "Invalid params", Data: "'ip_address' is required."}
	}

	// 2. Call the service layer function.
	details, err := GetIPDetails(ctx, ip)
	if err != nil {
		// 3. Translate the service layer error into a JSON-RPC error.
		return nil, &protocol.Error{Code: -32000, Message: "Failed to get IP details", Data: err.Error()}
	}

	// 4. Format the successful result for the MCP client.
	// We can choose to return a subset of the fields from the service layer.
	result := map[string]any{
		"ip":      details.Query,
		"country": details.Country,
		"city":    details.City,
		"isp":     details.ISP,
		"lat":     details.Lat,
		"lon":     details.Lon,
	}

	return result, nil
}

//======================================================================================
// LAYER 3: SERVER LAYER
// This layer is the main entry point. It defines and registers the MCP primitives.
//======================================================================================

func main() {
	// Create a new MCP server instance.
	s := server.NewMCPServer("IPGeolocationServer", "1.0.0")

	// Define the tool and its input schema.
	ipTool := protocol.NewTool(
		"ip/geolocate",
		"Gets geographic information for a given IP address.",
	).WithInputSchema(protocol.Object(
		protocol.Property("ip_address",
			protocol.String().WithDescription("The IPv4 or IPv6 address to locate."),
			protocol.Required(),
		),
	))

	// Register the tool with its handler from the tools layer.
	s.AddTool(ipTool, server.NewToolHandler(HandleIPGeolocate))

	// Start the server, listening on stdio.
	ctx := context.Background()
	log.Println("IP Geolocation MCP Server starting on stdio...")
	if err := server.ServeStdio(ctx, s); err != nil {
		log.Fatalf("Server exited with error: %v", err)
	}
	log.Println("Server shutting down.")
}

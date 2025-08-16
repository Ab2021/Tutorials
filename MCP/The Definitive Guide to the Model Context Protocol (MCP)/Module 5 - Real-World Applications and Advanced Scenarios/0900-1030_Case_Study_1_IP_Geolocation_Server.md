
# The Definitive Guide to the Model Context Protocol (MCP)

## Module 5: Real-World Applications and Advanced Scenarios

### Lesson 5.1: Case Study 1: The IP Geolocation Server (09:00 - 10:30)

---

### **1. Objective: Bridging MCP and the Public Web**

So far, our servers have operated on local data—files and directories. However, a vast amount of the world's information resides on the public internet, accessible via third-party APIs. This case study will demonstrate how to build a practical, self-contained MCP server that acts as a secure bridge between an MCP client and a public web service.

**Our Goal:** To build an MCP server that provides a tool to get geographic information for any given IP address. This tool, `ip/geolocate`, will take an IP address as input and return structured data including the country, city, and ISP of that IP.

**The Third-Party API:** We will use the free and simple-to-use [ip-api.com](http://ip-api.com/) service. It provides a RESTful API that returns JSON data, making it a perfect candidate for our integration.

**Key Learning Outcomes:**

*   **Architecting for External Services:** How to structure an MCP server with a clean separation of concerns, isolating the API interaction logic from the MCP protocol logic.
*   **Handling Network Operations:** Best practices for making HTTP requests within a tool handler.
*   **Data Transformation:** How to parse the JSON response from an external API and transform it into the structured output of an MCP tool.
*   **Error Handling:** How to gracefully handle potential errors from the external API (e.g., invalid input, network failures, API errors) and translate them into meaningful JSON-RPC errors.

This case study represents a significant step forward, moving from local utilities to network-aware, data-enriching services, a cornerstone of many real-world AI applications.

---

### **2. The Three-Layer Architecture**

A robust server that interacts with external services should be designed with a clear separation of concerns. For this case study, we will structure our Go application into three distinct layers. This is a common software engineering pattern that promotes modularity, testability, and maintainability.

**(ASCII Art Diagram of the Server Architecture)**

```
+--------------------------------------------------------------------+
|                           MCP Client                               |
+--------------------------------------------------------------------+
                   | (JSON-RPC over stdio/http)
                   v
+--------------------------------------------------------------------+
|                      main.go (The Server Layer)                    |
|--------------------------------------------------------------------|
| - Creates the MCP Server instance.                                 |
| - Defines the `ip/geolocate` tool and its schema.                  |
| - Registers the tool handler.                                      |
| - Starts the transport listener (e.g., `server.ServeStdio`).       |
| - This layer knows about MCP. It doesn't know about ip-api.com.    |
+--------------------------------------------------------------------+
                   | (Go function call)
                   v
+--------------------------------------------------------------------+
|                     tools.go (The Tools Layer)                     |
|--------------------------------------------------------------------|
| - Contains the `HandleIPGeolocate` tool handler function.          |
| - Parses and validates the input arguments from the MCP request.   |
| - Calls the appropriate service function.                          |
| - Formats the data from the service layer into the tool's output.  |
| - This layer acts as a bridge between MCP and the service.         |
+--------------------------------------------------------------------+
                   | (Go function call)
                   v
+--------------------------------------------------------------------+
|                    service.go (The Service Layer)                  |
|--------------------------------------------------------------------|
| - Contains the `GetIPDetails` function.                            |
| - Responsible for all interaction with the external ip-api.com.    |
| - Constructs the API URL, makes the HTTP GET request.              |
| - Parses the JSON response from the API into Go structs.           |
| - This layer knows nothing about MCP. It is pure business logic.   |
+--------------------------------------------------------------------+
```

**Why this architecture?**

*   **Modularity:** Each layer has a single, well-defined responsibility. The `service` layer can be tested independently of MCP. The `server` layer can be changed (e.g., from `stdio` to HTTP) without affecting the tool's logic.
*   **Testability:** It is easy to write unit tests for the `service` layer by mocking the HTTP requests. It's also easy to test the `tools` layer by providing mock service functions.
*   **Reusability:** The `service` layer could potentially be reused by other parts of a larger application that are not related to MCP.

---

### **3. Layer 1: The Service Layer (`service.go`)**

This is the innermost layer. Its only job is to communicate with the `ip-api.com` API. It knows nothing about MCP.

**The Data Structures:** First, we need Go structs that match the JSON response from the API. Looking at the [ip-api.com documentation](http://ip-api.com/docs/api:json), we can see the fields it returns.

```go
// service.go

// IPDetails corresponds to the JSON structure returned by the ip-api.com API.
// We use json tags to map the snake_case JSON fields to our CamelCase Go fields.
type IPDetails struct {
	Status      string  `json:"status"`
	Country     string  `json:"country"`
	City        string  `json:"city"`
	Lat         float64 `json:"lat"`
	Lon         float64 `json:"lon"`
	ISP         string  `json:"isp"`
	Query       string  `json:"query"`
    Message     string  `json:"message"` // Used for error responses from the API
}
```

**The Function:** Now, we write the function that takes an IP address, calls the API, and returns the parsed `IPDetails` struct.

```go
// service.go (continued)

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
)

const apiBaseURL = "http://ip-api.com/json/"

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
```

This layer is now a complete, self-contained, and testable package for interacting with the IP geolocation API.

---

### **4. Layer 2: The Tools Layer (`tools.go`)**

This layer acts as the bridge. It understands both the MCP protocol and the `service` layer.

**The Tool Handler:** Its job is to parse the MCP request, call the service function, and format the result into an MCP response.

```go
// tools.go

import (
	"context"
	"log"

	"github.com/sourcegraph/mcp-go/protocol"
    // Assuming service.go is in a package named 'ipservice'
    "github.com/your-username/ip-server/ipservice"
)

// HandleIPGeolocate is the handler for the `ip/geolocate` tool.
func HandleIPGeolocate(ctx context.Context, req *protocol.Request) (any, *protocol.Error) {
	log.Println("Handling request for ip/geolocate tool...")

	// 1. Parse and validate the input argument from the MCP request.
	params, _ := req.Params.(map[string]any)
	args, _ := params["arguments"].(map[string]any)
	ip, ok := args["ip_address"].(string)
	if !ok {
		return nil, &protocol.Error{Code: -32602, Message: "Invalid params", Data: "'ip_address' is required."}
	}

	// 2. Call the service layer function.
	details, err := ipservice.GetIPDetails(ctx, ip)
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
```

This handler cleanly separates the protocol-specific parsing from the core business logic, making the flow of data clear and easy to follow.

---

### **5. Layer 3: The Server Layer (`main.go`)**

This is the outermost layer, the entry point of our application. Its job is to set up the server, define the tool's schema, register the handler, and start the listener.

```go
// main.go

import (
	"context"
	"log"

	"github.com/sourcegraph/mcp-go/protocol"
	"github.com/sourcegraph/mcp-go/server"
    // Assuming tools.go is in a package named 'mcp_tools'
    "github.com/your-username/ip-server/mcp_tools"
)

func main() {
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
	s.AddTool(ipTool, server.NewToolHandler(mcp_tools.HandleIPGeolocate))

	// Start the server.
	ctx := context.Background()
	log.Println("IP Geolocation MCP Server starting on stdio...")
	if err := server.ServeStdio(ctx, s); err != nil {
		log.Fatalf("Server exited with error: %v", err)
	}
}
```

And with that, our server is complete. We have successfully built a modular, testable, and practical MCP server that integrates with a real-world, third-party API. This architectural pattern can be used as a blueprint for building much more complex and powerful MCP servers that connect to any number of internal or external services.

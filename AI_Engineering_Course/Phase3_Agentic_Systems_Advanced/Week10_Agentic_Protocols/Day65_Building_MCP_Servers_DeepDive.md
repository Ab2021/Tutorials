# Day 65: Building MCP Servers
## Deep Dive - Internal Mechanics & Advanced Reasoning

### TypeScript MCP Server (Node.js)

We'll build a "Weather MCP Server" using the TypeScript SDK.

**Setup:**
`npm install @modelcontextprotocol/sdk zod`

**Code (`index.ts`):**

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";

// 1. Create Server
const server = new McpServer({
  name: "weather-server",
  version: "1.0.0",
});

// 2. Define Tool
server.tool(
  "get-weather",
  "Get current weather for a city",
  {
    city: z.string().describe("The city name, e.g. San Francisco"),
    unit: z.enum(["C", "F"]).default("C").describe("Temperature unit"),
  },
  async ({ city, unit }) => {
    // Mock API call
    const temp = Math.floor(Math.random() * 30);
    return {
      content: [
        {
          type: "text",
          text: `Weather in ${city}: ${temp}°${unit}. Sunny.`,
        },
      ],
    };
  }
);

// 3. Define Resource
server.resource(
  "weather-alerts",
  "weather://alerts",
  async (uri) => {
    return {
      contents: [
        {
          uri: uri.href,
          text: "WARNING: Heatwave approaching.",
          mimeType: "text/plain",
        },
      ],
    };
  }
);

// 4. Connect Transport
async function main() {
  const transport = new StdioServerTransport();
  await server.connect(transport);
  console.error("Weather MCP Server running on stdio");
}

main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
```

### Python MCP Server (Advanced Patterns)

Using `FastMCP` for a "File System" server with Image Support.

```python
from mcp.server.fastmcp import FastMCP, Image
from pathlib import Path
import base64

mcp = FastMCP("FileServer")
ROOT_DIR = Path("./safe_zone")

@mcp.tool()
def list_files() -> str:
    """List files in the safe zone."""
    return "\n".join([p.name for p in ROOT_DIR.glob("*")])

@mcp.tool()
def read_image(filename: str) -> Image:
    """Read an image file and return it to the LLM."""
    path = ROOT_DIR / filename
    # Security Check
    if not path.is_relative_to(ROOT_DIR):
        raise ValueError("Access denied")
        
    if not path.exists():
        raise FileNotFoundError("File not found")
        
    with open(path, "rb") as f:
        data = f.read()
        
    return Image(data=data, format="png")

# Note: The 'Image' return type automatically handles 
# base64 encoding and MIME type setting in the JSON-RPC response.
```

### Debugging with the MCP Inspector

Anthropic provides an **MCP Inspector** tool to debug servers without Claude.

`npx @modelcontextprotocol/inspector node build/index.js`

This launches a web UI (localhost:5173) where you can:
1.  See the list of Tools/Resources.
2.  Manually invoke a tool and see the JSON response.
3.  View the raw JSON-RPC log.

### Handling Binary Data

MCP supports binary data (images, PDFs).
*   **Resources:** Return `blob` content (Base64 encoded) instead of `text`.
*   **Tools:** Can return `type: "image"` in the content list.
*   **Client Handling:** The Client (Claude) automatically renders this image in the chat UI or processes it with its vision model.

### Summary

*   **TypeScript:** Best for async I/O, web APIs.
*   **Python:** Best for local file manipulation, data processing.
*   **Inspector:** Your best friend for debugging.

# Day 64: Model Context Protocol (MCP) Fundamentals
## Deep Dive - Internal Mechanics & Advanced Reasoning

### The JSON-RPC Protocol

Under the hood, MCP uses **JSON-RPC 2.0**. It's a stateless, light-weight remote procedure call protocol.

**Request (Client -> Server):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {}
}
```

**Response (Server -> Client):**
```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "tools": [
      {
        "name": "calculate_sum",
        "description": "Adds two numbers",
        "inputSchema": {
          "type": "object",
          "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"}
          }
        }
      }
    ]
  }
}
```

### Implementing a Simple MCP Server (Python)

We will use the `mcp` python SDK to create a server that exposes a local SQLite database.

**Installation:**
`pip install mcp`

**Code (`server.py`):**

```python
from mcp.server.fastmcp import FastMCP
import sqlite3

# 1. Initialize Server
mcp = FastMCP("SQLite Explorer")

# 2. Define a Resource (Reading Data)
@mcp.resource("sqlite://{table_name}")
def read_table(table_name: str) -> str:
    """Read all rows from a table."""
    conn = sqlite3.connect("my_db.sqlite")
    cursor = conn.cursor()
    try:
        cursor.execute(f"SELECT * FROM {table_name} LIMIT 10")
        rows = cursor.fetchall()
        return str(rows)
    except Exception as e:
        return f"Error: {e}"
    finally:
        conn.close()

# 3. Define a Tool (Executing Action)
@mcp.tool()
def execute_query(query: str) -> str:
    """Execute a raw SQL query. Use with caution."""
    # Security: In prod, validate this heavily!
    if "DROP" in query.upper():
        return "Error: DROP not allowed."
        
    conn = sqlite3.connect("my_db.sqlite")
    cursor = conn.cursor()
    cursor.execute(query)
    conn.commit()
    return "Query executed successfully."

# 4. Run Server
if __name__ == "__main__":
    mcp.run()
```

### Configuring the Client (Claude Desktop)

To use this server, you edit the `claude_desktop_config.json` file.

```json
{
  "mcpServers": {
    "sqlite-explorer": {
      "command": "python",
      "args": ["/path/to/server.py"]
    }
  }
}
```

**What happens on startup:**
1.  Claude Desktop reads the config.
2.  It runs `python server.py`.
3.  It sends `initialize` handshake.
4.  It sends `tools/list` and `resources/list`.
5.  It injects the tool definitions into the System Prompt of the model.

### Advanced: Resource Templates

MCP supports dynamic resources using URI templates.
Instead of listing every single file, a server can say:
"I handle anything matching `file:///{path}`."

When the LLM asks for `file:///logs/app.log`, the Client parses the template, extracts `path="logs/app.log"`, and calls the Server's resource handler with that argument.

### Sampling (Context Management)

If a resource is huge (1GB log file), the Server can implement `sampling`.
The Client sends a `sampling/createMessage` request, asking the Server to "summarize" or "sample" the content using a smaller, local model (if the Server has one) or just truncating it, before sending it back to the main LLM.

### Summary

MCP is fundamentally about **Inversion of Control**. Instead of the Agent fetching data, the Agent *asks* the Server to provide data/tools, and the Server defines the boundaries of what is possible.

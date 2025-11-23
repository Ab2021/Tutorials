# Lab 1: Build an MCP Server

## Objective
Create a **Model Context Protocol (MCP)** server.
This server will expose a local SQLite database to any MCP-compliant client (like Claude Desktop or Cursor).

## 1. Setup

```bash
npm install @modelcontextprotocol/sdk
```

## 2. The Server (`server.ts`)

We will use TypeScript (standard for MCP).

```typescript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import sqlite3 from "sqlite3";

// 1. Database
const db = new sqlite3.Database("my_data.db");
db.run("CREATE TABLE IF NOT EXISTS notes (id INTEGER PRIMARY KEY, content TEXT)");
db.run("INSERT INTO notes (content) VALUES ('Buy milk'), ('Walk dog')");

// 2. Server
const server = new McpServer({
  name: "my-sqlite-server",
  version: "1.0.0",
});

// 3. Tool: Read Notes
server.tool(
  "read_notes",
  {},
  async () => {
    return new Promise((resolve) => {
      db.all("SELECT * FROM notes", (err, rows) => {
        resolve({
          content: [{ type: "text", text: JSON.stringify(rows) }],
        });
      });
    });
  }
);

// 4. Tool: Add Note
server.tool(
  "add_note",
  { content: z.string() },
  async ({ content }) => {
    db.run("INSERT INTO notes (content) VALUES (?)", [content]);
    return { content: [{ type: "text", text: "Note added." }] };
  }
);

// 5. Start
const transport = new StdioServerTransport();
await server.connect(transport);
```

## 3. Running the Lab

1.  Compile: `tsc server.ts`.
2.  Configure Claude Desktop:
    *   Edit `claude_desktop_config.json`.
    *   Add your server:
        ```json
        "mcpServers": {
          "my-sqlite": {
            "command": "node",
            "args": ["/path/to/server.js"]
          }
        }
        ```
3.  Restart Claude.
4.  Ask Claude: "Read my notes."

## 4. Submission
Submit a screenshot of Claude accessing your local database.

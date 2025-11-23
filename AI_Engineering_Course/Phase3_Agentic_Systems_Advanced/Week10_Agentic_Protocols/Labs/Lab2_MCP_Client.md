# Lab 2: Build an MCP Client

## Objective
Build a Python client that connects to the MCP server you built in Lab 1.
This shows how to consume MCP tools programmatically.

## 1. Setup

```bash
pip install mcp
```

## 2. The Client (`client.py`)

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def run():
    # 1. Configure Connection
    server_params = StdioServerParameters(
        command="node",
        args=["/path/to/server.js"], # Path from Lab 1
    )

    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. Initialize
            await session.initialize()
            
            # 3. List Tools
            tools = await session.list_tools()
            print("Available Tools:", [t.name for t in tools.tools])
            
            # 4. Call Tool
            result = await session.call_tool("read_notes", arguments={})
            print("Notes:", result.content[0].text)
            
            # 5. Add Note
            await session.call_tool("add_note", arguments={"content": "Learn MCP"})
            print("Note added.")

if __name__ == "__main__":
    asyncio.run(run())
```

## 3. Analysis
You have now built a **Tool Use Bridge**.
Your Python script can control any resource exposed via MCP (Databases, APIs, Filesystems).

## 4. Submission
Submit the console output.

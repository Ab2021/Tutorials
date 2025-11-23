# Lab 5: MCP Inspector

## Objective
Debug your MCP servers.
Build a simple client that lists tools and resources.

## 1. The Inspector (`inspector.py`)

```python
import asyncio
from mcp.client import Client

async def inspect():
    # Connect to a local server (mock connection)
    async with Client("stdio_connection") as client:
        print("--- Tools ---")
        tools = await client.list_tools()
        for t in tools:
            print(f"- {t.name}: {t.description}")
            
        print("\n--- Resources ---")
        resources = await client.list_resources()
        for r in resources:
            print(f"- {r.name} ({r.uri})")

# asyncio.run(inspect())
```

## 2. Challenge
Add a **Tool Tester**. Allow the user to select a tool and execute it with inputs.

## 3. Submission
Submit the output of the inspector running against your Lab 1 server.

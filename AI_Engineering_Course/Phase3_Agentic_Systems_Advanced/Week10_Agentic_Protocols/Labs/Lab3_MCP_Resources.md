# Lab 3: MCP Resource Server

## Objective
Expose data to LLMs using **MCP Resources**.
Resources are passive data sources (files, logs, DB rows).

## 1. The Server (`resources.py`)

```python
from mcp.server import Server
from mcp.types import Resource

app = Server("MyResources")

@app.list_resources()
async def list_resources():
    return [
        Resource(uri="file:///logs/app.log", name="Application Logs", mimeType="text/plain")
    ]

@app.read_resource()
async def read_resource(uri):
    if uri == "file:///logs/app.log":
        return "Error: Connection failed at 10:00 AM"
    return "Not Found"

# Run using mcp-server-runner (CLI)
```

## 2. Analysis
Resources allow the LLM to "read" your system without you pasting text into the prompt.

## 3. Submission
Submit the `read_resource` function code.

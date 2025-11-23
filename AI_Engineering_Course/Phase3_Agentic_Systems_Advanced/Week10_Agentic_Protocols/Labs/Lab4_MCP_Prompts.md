# Lab 4: MCP Prompts Server

## Objective
Share **Prompt Templates** across clients using MCP.
Standardize how your team talks to the LLM.

## 1. The Server (`prompts.py`)

```python
from mcp.server import Server
from mcp.types import Prompt, PromptArgument

app = Server("MyPrompts")

@app.list_prompts()
async def list_prompts():
    return [
        Prompt(
            name="code_review",
            description="Review code for bugs",
            arguments=[PromptArgument(name="code", required=True)]
        )
    ]

@app.get_prompt()
async def get_prompt(name, arguments):
    if name == "code_review":
        code = arguments['code']
        return {
            "messages": [
                {"role": "user", "content": f"Review this code:\n{code}"}
            ]
        }
```

## 2. Analysis
Clients (Claude, Cursor) can now see "code_review" in their slash commands.

## 3. Submission
Submit the JSON response for a `get_prompt` call.

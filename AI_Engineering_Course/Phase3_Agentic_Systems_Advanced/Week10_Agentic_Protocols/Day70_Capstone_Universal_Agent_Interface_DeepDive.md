# Day 70: Capstone: Building a Universal Agent Interface
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing the UAI Kernel

We will build a simplified Kernel in Python that bridges MCP and Agent Protocol.

**Dependencies:**
`pip install mcp agent-protocol openai`

**Code (`uai_kernel.py`):**

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
import requests
from openai import AsyncOpenAI

class UniversalAgent:
    def __init__(self, name):
        self.name = name
        self.llm = AsyncOpenAI()
        self.mcp_tools = []
        
    async def load_mcp_server(self, command, args):
        """Connect to a local MCP Server."""
        params = StdioServerParameters(command=command, args=args)
        self.mcp_ctx = stdio_client(params)
        self.read, self.write = await self.mcp_ctx.__aenter__()
        self.session = ClientSession(self.read, self.write)
        await self.session.initialize()
        
        # Load Tools
        tools = await self.session.list_tools()
        for t in tools.tools:
            self.mcp_tools.append({
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema
                }
            })
        print(f"Loaded MCP Tools: {[t['function']['name'] for t in self.mcp_tools]}")

    async def delegate_to_agent(self, agent_url, task_input):
        """Delegate to a remote Agent Protocol agent."""
        print(f"Delegating to {agent_url}...")
        # 1. Create Task
        resp = requests.post(f"{agent_url}/agent/tasks", json={"input": task_input})
        task_id = resp.json()["task_id"]
        
        # 2. Poll for completion
        while True:
            step_resp = requests.post(f"{agent_url}/agent/tasks/{task_id}/steps")
            step = step_resp.json()
            if step["is_last"]:
                return step["output"]
            await asyncio.sleep(1)

    async def run(self, user_goal):
        """Main Execution Loop."""
        messages = [{"role": "user", "content": user_goal}]
        
        # 1. Decide: Local Tool or Remote Agent?
        # We inject a special "delegate" tool into the LLM's definition
        all_tools = self.mcp_tools + [{
            "type": "function",
            "function": {
                "name": "delegate_task",
                "description": "Delegate a subtask to a remote specialist agent.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "agent_url": {"type": "string"},
                        "task": {"type": "string"}
                    }
                }
            }
        }]
        
        response = await self.llm.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=all_tools
        )
        
        msg = response.choices[0].message
        
        if msg.tool_calls:
            for tc in msg.tool_calls:
                fn = tc.function.name
                args = json.loads(tc.function.arguments)
                
                if fn == "delegate_task":
                    result = await self.delegate_to_agent(args["agent_url"], args["task"])
                else:
                    # MCP Tool
                    result = await self.session.call_tool(fn, arguments=args)
                    result = result.content[0].text
                    
                print(f"Tool/Agent Result: {result}")

# Usage
async def main():
    uai = UniversalAgent("Orchestrator")
    await uai.load_mcp_server("python", ["weather_server.py"])
    
    # User asks for something that requires both local weather and remote writing
    await uai.run("Check the weather in NY and ask the writer agent (http://localhost:8000/ap/v1) to write a poem about it.")

if __name__ == "__main__":
    asyncio.run(main())
```

### The "Glue" Logic

The key innovation here is treating **Remote Agents** as just another **Tool** in the LLM's context.
*   Local Tool: `call_tool(args)` -> MCP
*   Remote Agent: `delegate_task(url, task)` -> Agent Protocol

### Handling Identity

To make this secure, we would wrap the `requests.post` call in `delegate_to_agent` with a DID signature header.
`headers={"X-Agent-Signature": self.identity.sign(payload)}`

### Summary

The UAI is the convergence point. It doesn't care *where* the capability comes from (local process, remote server, blockchain). It just orchestrates them to satisfy the user's intent.

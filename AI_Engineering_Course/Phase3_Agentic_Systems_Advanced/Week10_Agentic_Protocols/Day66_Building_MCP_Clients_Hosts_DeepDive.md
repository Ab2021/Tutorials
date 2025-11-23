# Day 66: Building MCP Clients & Hosts
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building a Custom MCP Client (Python)

We will build a simple CLI Chatbot that can use any MCP Server.

**Dependencies:**
`pip install mcp openai`

**Code (`client.py`):**

```python
import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# 1. Configuration (Connect to a Server)
server_params = StdioServerParameters(
    command="python",
    args=["weather_server.py"], # The server we built yesterday
)

async def run_agent():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # 2. Initialize
            await session.initialize()
            
            # 3. List Tools
            tools_response = await session.list_tools()
            openai_tools = []
            
            for tool in tools_response.tools:
                openai_tools.append({
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.inputSchema
                    }
                })
            
            print(f"Loaded {len(openai_tools)} tools.")

            # 4. Chat Loop
            client = AsyncOpenAI()
            messages = [{"role": "user", "content": "What is the weather in Tokyo?"}]
            
            response = await client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                tools=openai_tools
            )
            
            msg = response.choices[0].message
            print(f"Agent: {msg.content}")
            
            # 5. Handle Tool Call
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    fn_name = tc.function.name
                    args = tc.function.arguments
                    print(f"Calling Tool: {fn_name}({args})")
                    
                    # Call MCP Server
                    result = await session.call_tool(fn_name, arguments=json.loads(args))
                    
                    # Feed back to LLM
                    messages.append(msg)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": str(result.content)
                    })
                    
                    # Get final answer
                    final = await client.chat.completions.create(
                        model="gpt-4o",
                        messages=messages
                    )
                    print(f"Final: {final.choices[0].message.content}")

if __name__ == "__main__":
    asyncio.run(run_agent())
```

### Managing Multiple Servers

To handle multiple servers, you need a **Connection Manager**.

```python
class ConnectionManager:
    def __init__(self):
        self.sessions = {} # server_name -> session

    async def connect(self, name, command, args):
        # ... setup stdio_client ...
        self.sessions[name] = session
        
    async def get_all_tools(self):
        all_tools = []
        for name, session in self.sessions.items():
            tools = await session.list_tools()
            # Namespace the tools to avoid collisions
            # e.g. "weather_server__get_weather"
            for t in tools.tools:
                t.name = f"{name}__{t.name}" 
                all_tools.append(t)
        return all_tools
        
    async def route_tool_call(self, tool_name, args):
        server_name, real_tool_name = tool_name.split("__")
        session = self.sessions[server_name]
        return await session.call_tool(real_tool_name, args)
```

### Handling Resources

Resources are simpler. You usually fetch them *before* the chat or on demand.

```python
# Fetch a resource
resource = await session.read_resource("file:///logs/error.txt")
content = resource.contents[0].text

# Inject into System Prompt
system_prompt = f"""
You have access to the following log file:
---
{content}
---
"""
```

### Summary

Building a Client is about **Translation**. You translate MCP Tool definitions into OpenAI/Anthropic Tool definitions, and you translate OpenAI Tool Calls into MCP `call_tool` requests. The Client acts as the bridge.

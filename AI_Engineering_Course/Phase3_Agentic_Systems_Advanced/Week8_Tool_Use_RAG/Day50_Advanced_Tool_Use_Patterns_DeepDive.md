# Day 50: Advanced Tool Use Patterns
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Building a Robust Tool Execution Engine

In this deep dive, we will move beyond the basic OpenAI SDK examples and build a production-grade tool executor. We will handle:
1.  **Pydantic Validation:** Ensuring arguments match the schema.
2.  **Error Handling:** Feeding errors back to the model so it can self-correct.
3.  **Parallel Execution:** Running independent tools concurrently.

### 1. Defining Tools with Pydantic

Using Pydantic allows us to auto-generate the JSON schema required by OpenAI/Anthropic.

```python
import json
from typing import List, Optional
from pydantic import BaseModel, Field
import inspect

class ToolBase(BaseModel):
    @classmethod
    def to_openai_schema(cls):
        return {
            "name": cls.__name__,
            "description": cls.__doc__,
            "parameters": cls.model_json_schema()
        }

class GetWeather(ToolBase):
    """Get the current weather for a specific location."""
    location: str = Field(..., description="The city and state, e.g. San Francisco, CA")
    unit: str = Field("celsius", enum=["celsius", "fahrenheit"], description="The temperature unit")

class SearchWeb(ToolBase):
    """Search the internet for up-to-date information."""
    query: str = Field(..., description="The search query")

# Registry of available tools
TOOLS = [GetWeather, SearchWeb]
TOOL_SCHEMAS = [t.to_openai_schema() for t in TOOLS]
TOOL_MAP = {t.__name__: t for t in TOOLS}
```

### 2. The Execution Loop

This is the heart of the agent. It needs to handle the model's response, execute tools, and manage the conversation history.

```python
from openai import OpenAI
import time

client = OpenAI()

def execute_tool_call(tool_call):
    """Executes a single tool call safely."""
    function_name = tool_call.function.name
    arguments_str = tool_call.function.arguments
    
    print(f"🛠️ Model requested: {function_name}({arguments_str})")
    
    if function_name not in TOOL_MAP:
        return {"error": f"Tool {function_name} not found."}
    
    tool_class = TOOL_MAP[function_name]
    
    try:
        # 1. Validate Arguments using Pydantic
        args_dict = json.loads(arguments_str)
        tool_instance = tool_class(**args_dict)
        
        # 2. Execute Logic (Mocked here)
        if isinstance(tool_instance, GetWeather):
            return {"temperature": 22, "unit": tool_instance.unit, "condition": "Sunny"}
        elif isinstance(tool_instance, SearchWeb):
            return {"results": ["Python 3.12 released...", "New AI models..."]}
            
    except json.JSONDecodeError:
        return {"error": "Invalid JSON arguments provided."}
    except Exception as e:
        return {"error": f"Tool execution failed: {str(e)}"}

def run_agent(user_query):
    messages = [{"role": "user", "content": user_query}]
    
    while True:
        # 1. Call Model
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            tools=[{"type": "function", "function": s} for s in TOOL_SCHEMAS],
            tool_choice="auto"
        )
        
        msg = response.choices[0].message
        messages.append(msg)
        
        # 2. Check for Tool Calls
        if not msg.tool_calls:
            print("🤖 Final Answer:", msg.content)
            break
            
        # 3. Handle Tool Calls (Parallel Support)
        for tool_call in msg.tool_calls:
            # Execute
            result = execute_tool_call(tool_call)
            
            # Append Result
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
            })
            
        # Loop continues to let model process the tool results
```

### 3. Advanced Pattern: Error Recovery

What happens when the model generates invalid arguments?
If we just crash, the user experience is terrible.
Instead, we feed the error back to the model.

*   **Scenario:** Model calls `GetWeather(location="Tokyo", unit="kelvin")`.
*   **Pydantic Error:** `Input should be 'celsius' or 'fahrenheit'`.
*   **System Action:** Send this error message as the tool result.
*   **Model Reaction:** The model sees the error, realizes its mistake, and re-calls the tool with `unit="celsius"`.

This "Self-Correction" loop is what makes agents robust.

### 4. Advanced Pattern: Dynamic Tool Selection

If you have 100 tools, you can't put them all in the context window.
**Solution:** Retrieval-Augmented Tool Use.

1.  **Embed** descriptions of all 100 tools.
2.  **Embed** the user's query.
3.  **Retrieve** the top 5 most relevant tools.
4.  **Inject** only those 5 schemas into the API call.

```python
# Conceptual Implementation
def get_relevant_tools(query, all_tools):
    # Vector search logic here...
    return top_5_tools

# In the loop:
relevant_tools = get_relevant_tools(user_query, ALL_TOOLS)
response = client.chat.completions.create(..., tools=relevant_tools)
```

### 5. Security: The Sandboxed Executor

Never run `exec()` or `os.system()` directly.
For code execution tools (like a Python REPL), use a sandbox:
*   **Docker:** Spin up an ephemeral container for each request.
*   **gVisor:** Google's container runtime sandbox.
*   **e2b:** A managed infrastructure for AI code execution.

```python
# Unsafe
def run_python(code):
    exec(code) # DANGER!

# Safe(r)
def run_python_safe(code):
    container = docker.run("python:3.9-slim", command=["python", "-c", code])
    return container.logs()
```

### Summary

Building a tool-using agent is more than just an API call. It involves strict validation, graceful error handling loops, and security considerations. By treating tools as typed Pydantic models, we gain robustness and clarity in our agent's "interface" to the world.

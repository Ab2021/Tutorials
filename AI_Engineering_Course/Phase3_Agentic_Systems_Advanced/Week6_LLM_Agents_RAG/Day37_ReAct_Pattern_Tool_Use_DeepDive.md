# Day 37: ReAct Pattern & Tool Use
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. ReAct Prompt Optimization

**Minimal Prompt (Baseline):**
```
Use tools to answer: {question}
Tools: search, calculate
```
**Performance:** 60% success rate.

**Optimized Prompt:**
```
You are a helpful assistant with access to tools.

Think step-by-step using this format:
Thought: <your reasoning>
Action: <tool>[<input>]
Observation: <result>
...
Answer: <final answer>

Tools:
- search[query]: Search the web
- calculate[expr]: Evaluate math

Question: {question}
Thought:
```
**Performance:** 85% success rate (+25%).

**Key Improvements:**
- Explicit format specification.
- Tool descriptions.
- Starting with "Thought:" primes the model.

### 2. Tool Execution Engine

**Synchronous Execution:**
```python
def execute_tool(tool_name, tool_input):
    if tool_name == "search":
        return web_search(tool_input)
    elif tool_name == "calculate":
        return eval(tool_input)
    else:
        return f"Error: Unknown tool {tool_name}"
```

**Asynchronous Execution:**
```python
import asyncio

async def execute_tool_async(tool_name, tool_input):
    if tool_name == "search":
        return await web_search_async(tool_input)
    elif tool_name == "calculate":
        return await calculate_async(tool_input)
    else:
        return f"Error: Unknown tool {tool_name}"

# Parallel execution
results = await asyncio.gather(
    execute_tool_async("search", "Paris"),
    execute_tool_async("search", "London")
)
```

### 3. Advanced Parsing Techniques

**Regex-Based Parsing:**
```python
import re

def parse_action(text):
    # Match: Action: tool_name[input]
    pattern = r'Action\s*\d*:\s*(\w+)\[(.*?)\]'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        return {
            "tool": match.group(1),
            "input": match.group(2).strip()
        }
    return None
```

**Structured Output (Function Calling):**
```python
# OpenAI Function Calling
response = openai.ChatCompletion.create(
    model="gpt-4",
    messages=messages,
    functions=[{
        "name": "search",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            }
        }
    }],
    function_call="auto"
)

if response.choices[0].message.get("function_call"):
    function_call = response.choices[0].message.function_call
    tool_name = function_call.name
    tool_input = json.loads(function_call.arguments)
```

### 4. Error Recovery Strategies

**Retry with Reflection:**
```python
def execute_with_retry(agent, question, max_retries=3):
    for attempt in range(max_retries):
        try:
            result = agent.run(question)
            return result
        except ToolExecutionError as e:
            if attempt < max_retries - 1:
                # Add error to context
                agent.add_message(f"Error: {e}. Please try a different approach.")
            else:
                return f"Failed after {max_retries} attempts: {e}"
```

**Fallback Chain:**
```python
def search_with_fallback(query):
    # Try primary search
    try:
        return google_search(query)
    except:
        pass
    
    # Fallback to Bing
    try:
        return bing_search(query)
    except:
        pass
    
    # Fallback to DuckDuckGo
    try:
        return duckduckgo_search(query)
    except:
        return "All search engines failed"
```

### 5. Complete Production ReAct Agent

```python
import openai
import re
import time
from typing import List, Dict, Callable

class ProductionReActAgent:
    def __init__(
        self,
        tools: Dict[str, Callable],
        model: str = "gpt-4",
        max_iterations: int = 10,
        timeout: int = 60
    ):
        self.tools = tools
        self.model = model
        self.max_iterations = max_iterations
        self.timeout = timeout
        self.start_time = None
    
    def run(self, question: str) -> str:
        self.start_time = time.time()
        messages = self._initialize_messages(question)
        
        for iteration in range(self.max_iterations):
            # Check timeout
            if time.time() - self.start_time > self.timeout:
                return "Timeout: Task took too long"
            
            # Get model response
            try:
                response = self._get_model_response(messages)
            except Exception as e:
                return f"Model error: {e}"
            
            content = response.choices[0].message.content
            messages.append({"role": "assistant", "content": content})
            
            # Check if finished
            if self._is_finished(content):
                return self._extract_answer(content)
            
            # Parse and execute action
            action = self._parse_action(content)
            if action is None:
                messages.append({
                    "role": "user",
                    "content": "Error: Could not parse action. Please use format: Action: tool[input]"
                })
                continue
            
            # Execute tool with error handling
            observation = self._execute_tool(action["tool"], action["input"])
            messages.append({
                "role": "user",
                "content": f"Observation: {observation}"
            })
        
        return "Max iterations reached without finding answer"
    
    def _initialize_messages(self, question: str) -> List[Dict]:
        system_prompt = self._get_system_prompt()
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Question: {question}\nThought:"}
        ]
    
    def _get_system_prompt(self) -> str:
        tool_descriptions = "\n".join([
            f"- {name}: {func.__doc__ or 'No description'}"
            for name, func in self.tools.items()
        ])
        
        return f"""You are a helpful assistant with access to tools.

Think step-by-step using this format:
Thought: <reasoning>
Action: <tool>[<input>]
(wait for Observation)
Thought: <reasoning>
...
Answer: <final answer>

Available tools:
{tool_descriptions}

Always start with Thought. Use Answer when you have the final response."""
    
    def _get_model_response(self, messages: List[Dict]):
        return openai.ChatCompletion.create(
            model=self.model,
            messages=messages,
            temperature=0,
            max_tokens=500
        )
    
    def _is_finished(self, content: str) -> bool:
        return "Answer:" in content or "answer:" in content.lower()
    
    def _extract_answer(self, content: str) -> str:
        match = re.search(r'Answer:\s*(.+)', content, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return content
    
    def _parse_action(self, content: str) -> Dict:
        pattern = r'Action\s*\d*:\s*(\w+)\[(.*?)\]'
        match = re.search(pattern, content, re.IGNORECASE)
        
        if match:
            return {
                "tool": match.group(1),
                "input": match.group(2).strip()
            }
        return None
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        if tool_name not in self.tools:
            return f"Error: Unknown tool '{tool_name}'. Available: {list(self.tools.keys())}"
        
        try:
            result = self.tools[tool_name](tool_input)
            return str(result)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

# Example tools
def search(query: str) -> str:
    """Search the web for information"""
    # Mock implementation
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate a mathematical expression"""
    try:
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Calculation error: {e}"

# Usage
agent = ProductionReActAgent(
    tools={"search": search, "calculate": calculate},
    max_iterations=10,
    timeout=60
)

answer = agent.run("What is 15% of 240?")
print(answer)
```

### 6. Optimization: Tool Result Caching

```python
from functools import lru_cache
import hashlib

class CachedToolExecutor:
    def __init__(self):
        self.cache = {}
    
    def execute(self, tool_name: str, tool_input: str) -> str:
        # Create cache key
        cache_key = hashlib.md5(
            f"{tool_name}:{tool_input}".encode()
        ).hexdigest()
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Execute and cache
        result = self._execute_tool(tool_name, tool_input)
        self.cache[cache_key] = result
        return result
```

### 7. Metrics and Monitoring

```python
class InstrumentedReActAgent(ProductionReActAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.metrics = {
            "total_iterations": 0,
            "tool_calls": {},
            "errors": 0,
            "success": 0
        }
    
    def _execute_tool(self, tool_name: str, tool_input: str) -> str:
        self.metrics["total_iterations"] += 1
        self.metrics["tool_calls"][tool_name] = \
            self.metrics["tool_calls"].get(tool_name, 0) + 1
        
        result = super()._execute_tool(tool_name, tool_input)
        
        if "Error" in result:
            self.metrics["errors"] += 1
        
        return result
    
    def get_metrics(self) -> Dict:
        return self.metrics
```

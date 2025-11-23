# Day 50: Advanced Tool Use Patterns
## Core Concepts & Theory

### The Evolution of Tool Use

In the early days of LLMs (GPT-3), "tool use" was a hack. We used few-shot prompting to teach the model to output a specific string like `[SEARCH: "query"]`, which we then parsed with regex. This was brittle, error-prone, and consumed valuable context window tokens.

Today, **Tool Use (or Function Calling)** is a first-class citizen in modern model architectures. Models like GPT-4, Claude 3, and Llama 3 are fine-tuned specifically to detect when a user's request requires external data or actions, and to output a structured representation (usually JSON) of the function call.

### 1. The Function Calling Lifecycle

Understanding the flow is critical for debugging:

1.  **Definition:** The developer provides a list of available tools (schemas) in the API call.
    *   *Example:* `get_weather(location: str, unit: str)`
2.  **Reasoning:** The model analyzes the user prompt ("What's the weather in Tokyo?"). It determines that it cannot answer from its internal weights but *can* answer using the provided tool.
3.  **Generation:** The model generates a structured output: `{"name": "get_weather", "arguments": "{\"location\": \"Tokyo\", \"unit\": \"celsius\"}"}`.
    *   *Note:* The model pauses generation here. It does *not* execute the tool.
4.  **Execution:** The application runtime (your Python/JS code) parses this JSON, calls the actual API, and gets the result ("22°C").
5.  **Response:** The application sends the tool result back to the model as a new message with role `tool`.
6.  **Final Answer:** The model uses the tool result to generate the final natural language response to the user.

### 2. JSON Mode vs. Function Calling

While related, these are distinct capabilities:

*   **JSON Mode:** Forces the model to output valid JSON. Useful for extraction tasks (e.g., "Extract all dates from this text"). It doesn't necessarily imply calling an external tool.
*   **Function Calling:** A specific protocol where the model selects a tool from a provided list. It handles the "routing" logic (which tool to call?) in addition to the formatting logic.

### 3. Advanced Patterns

#### A. Parallel Function Calling
Modern models can call multiple tools in a single turn.
*   *User:* "Get the weather for Tokyo and New York."
*   *Model:* `[call_tool("weather", "Tokyo"), call_tool("weather", "NY")]`
*   This reduces latency significantly compared to sequential round-trips.

#### B. Nested Tool Calls (Chain of Thought with Tools)
Sometimes the output of Tool A is the input for Tool B.
*   *User:* "Email the report to the CEO."
*   *Step 1:* Call `get_employee_email(name="CEO")` -> Returns "ceo@company.com".
*   *Step 2:* Call `send_email(to="ceo@company.com", body="...")`.
*   *Orchestration:* This requires a loop in your application code to handle multi-step dependencies.

#### C. Tool Hallucination & Validation
Models can hallucinate tool names or arguments.
*   *Hallucinated Tool:* Calling `get_stock_price` when only `get_weather` was provided.
*   *Hallucinated Arg:* Calling `get_weather(city="Tokyo", date="tomorrow")` when the API doesn't support a date parameter.
*   *Mitigation:* Strict Pydantic validation of arguments before execution.

### 4. Schema Engineering

Just as Prompt Engineering optimizes text, **Schema Engineering** optimizes tool definitions.
*   **Descriptions Matter:** The docstring of the function is the "prompt" for the tool. Be verbose. "Calculates the square root" is better than "Math function".
*   **Type Hints:** Use Enums for categorical variables to restrict the model's output space.
*   **Simplicity:** Don't overload a single function with 50 optional parameters. Break it down into smaller, specific tools.

### 5. The "System Prompt" for Tools

Even with native function calling, the System Prompt plays a huge role.
*   *Instruction:* "You are a helpful assistant. Don't guess values. If a tool requires a parameter you don't have, ask the user for it."
*   *Constraint:* "Only use the `delete_user` tool if the user explicitly confirms."

### 6. Security Implications

Giving an LLM access to tools expands the attack surface.
*   **Prompt Injection:** An attacker can trick the model into calling `delete_database()`.
*   **SSRF (Server-Side Request Forgery):** If a tool takes a URL as input, an attacker could force the server to scan internal ports.
*   **Defense:** Always validate inputs. Implement "Human-in-the-loop" for sensitive actions. Use read-only API keys where possible.

### Summary

Tool Use transforms LLMs from passive text generators into active agents. Mastering it requires understanding the request-response lifecycle, designing clean API schemas, and implementing robust error handling for the inevitable model mistakes. In the Deep Dive, we will build a robust tool execution engine from scratch.

# Lab 2: Function Calling API

## Objective
Expose Python functions to OpenAI's GPT-4 using the `tools` parameter.
This is how ChatGPT Plugins work.

## 1. The Functions

```python
import json

def get_current_weather(location, unit="fahrenheit"):
    """Get the current weather in a given location"""
    if "tokyo" in location.lower():
        return json.dumps({"location": "Tokyo", "temperature": "10", "unit": "celsius"})
    elif "san francisco" in location.lower():
        return json.dumps({"location": "San Francisco", "temperature": "72", "unit": "fahrenheit"})
    else:
        return json.dumps({"location": location, "temperature": "unknown"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]
```

## 2. The Call (`call.py`)

```python
from openai import OpenAI
client = OpenAI()

messages = [{"role": "user", "content": "What's the weather like in San Francisco and Tokyo?"}]

response = client.chat.completions.create(
    model="gpt-3.5-turbo",
    messages=messages,
    tools=tools,
    tool_choice="auto", 
)

response_message = response.choices[0].message
tool_calls = response_message.tool_calls

if tool_calls:
    # Step 3: Call the function
    available_functions = {
        "get_current_weather": get_current_weather,
    } 
    messages.append(response_message)  # extend conversation with assistant's reply
    
    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions[function_name]
        function_args = json.loads(tool_call.function.arguments)
        
        function_response = function_to_call(
            location=function_args.get("location"),
            unit=function_args.get("unit"),
        )
        
        messages.append(
            {
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": function_name,
                "content": function_response,
            }
        )
        
    # Step 4: Get final response
    second_response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages,
    )
    print(second_response.choices[0].message.content)
```

## 3. Analysis
Notice how the model called the function **twice** (once for SF, once for Tokyo) in parallel (if using GPT-4) or sequentially.
It then synthesized the JSON outputs into a natural language sentence.

## 4. Submission
Submit the final output text.

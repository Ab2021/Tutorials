# Day 76: Fine-tuning Agents & Function Calling
## Deep Dive - Internal Mechanics & Advanced Reasoning

### 1. Synthetic Data Generation for Tools

Using GPT-4 to create training data for Llama-3.

```python
import json

tools = [
    {
        "name": "get_stock_price",
        "description": "Get current stock price",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}}
    }
]

def generate_example(llm):
    prompt = f"""
    Tools: {json.dumps(tools)}
    
    Generate a conversation where the user asks a question and the assistant calls a tool.
    Format: JSON {{user: "...", tool_call: "..."}}
    """
    # response = llm.generate(prompt)
    return {
        "user": "What is Apple trading at?",
        "tool_call": {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}
    }

# Generate 1000 examples -> Save to jsonl for training
```

### 2. Grammar-Constrained Decoding (llama-cpp-python)

Forcing valid JSON output.

```python
from llama_cpp import Llama, LlamaGrammar

# 1. Define Grammar (GBNF)
json_grammar = r"""
root   ::= object
object ::= "{" space ( member ("," space member)* )? space "}"
member ::= string ":" space value
string ::= "\"" ([^"]*) "\""
value  ::= string | number | object | array | "true" | "false" | "null"
space  ::= " "?
"""
# (Simplified GBNF)

grammar = LlamaGrammar.from_string(json_grammar)

# 2. Load Model
llm = Llama(model_path="llama-3-8b.Q4_K_M.gguf")

# 3. Generate
output = llm(
    "Generate a JSON object for a person:",
    grammar=grammar,
    max_tokens=100
)

print(output['choices'][0]['text'])
# Guaranteed to be valid JSON
```

### 3. Fine-tuning Format (ChatML)

Formatting data for training.

```python
def format_for_training(example):
    # ChatML format
    text = f"<|im_start|>user\n{example['user']}<|im_end|>\n"
    text += f"<|im_start|>assistant\n"
    text += f"<tool_code>{json.dumps(example['tool_call'])}</tool_code><|im_end|>"
    return text

# Dataset = [format_for_training(ex) for ex in examples]
# Train using HuggingFace TRL (SFTTrainer)
```

### 4. Function Calling Inference Loop

Executing the tool calls.

```python
def execute_tool_call(tool_call_json):
    name = tool_call_json['name']
    args = tool_call_json['arguments']
    
    if name == "get_stock_price":
        return f"Price of {args['symbol']} is $150"
    return "Unknown tool"

def agent_loop(llm, user_query):
    # 1. Generate Tool Call
    response = llm.generate(user_query)
    
    # 2. Parse
    try:
        tool_call = json.loads(response)
    except:
        return response # Normal text
        
    # 3. Execute
    result = execute_tool_call(tool_call)
    
    # 4. Generate Final Answer
    final = llm.generate(f"User: {user_query}\nTool Result: {result}\nAnswer:")
    return final
```

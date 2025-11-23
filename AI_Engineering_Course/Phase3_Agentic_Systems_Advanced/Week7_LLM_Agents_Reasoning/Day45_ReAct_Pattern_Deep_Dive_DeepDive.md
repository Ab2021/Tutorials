# Day 45: ReAct Pattern Deep Dive (Reasoning + Acting)
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a ReAct Loop from Scratch

We won't use LangChain. We'll build the raw loop to understand it.

```python
import re

class ReActAgent:
    def __init__(self, client, tools):
        self.client = client
        self.tools = tools # Dict of functions
        self.history = ""
        self.max_steps = 10

    def run(self, question):
        self.history = f"Question: {question}\n"
        
        for i in range(self.max_steps):
            # 1. Generate Thought & Action
            prompt = f"""
            Answer the following question using the available tools: {list(self.tools.keys())}
            Format:
            Thought: ...
            Action: ToolName(args)
            Observation: ...
            
            History:
            {self.history}
            """
            
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[{"role": "user", "content": prompt}],
                stop=["Observation:"] # Stop before hallucinating observation
            ).choices[0].message.content
            
            self.history += response + "\n"
            print(f"Step {i}: {response}")
            
            # 2. Parse Action
            if "Action:" in response:
                action_line = response.split("Action:")[-1].strip()
                tool_name, args = self.parse_action(action_line)
                
                # 3. Execute Tool
                if tool_name in self.tools:
                    observation = self.tools[tool_name](args)
                else:
                    observation = f"Error: Tool {tool_name} not found."
                    
                self.history += f"Observation: {observation}\n"
                
            elif "Answer:" in response:
                return response.split("Answer:")[-1].strip()
                
        return "Timeout"

    def parse_action(self, action_str):
        # Regex to extract Name(Args)
        match = re.match(r"(\w+)\((.*)\)", action_str)
        if match:
            return match.group(1), match.group(2)
        return None, None

# Mock Tools
def search(query):
    return "Tim Cook is 63 years old."

agent = ReActAgent(client, {"search": search})
agent.run("How old is Apple's CEO?")
```

### Prompt Engineering for ReAct

The Prompt is critical. It usually contains **Few-Shot Examples** of ReAct traces.
*   *Example 1:* Question: ... Thought: ... Action: ... Observation: ... Answer: ...
*   *Example 2:* ...
*   *Current:* Question: ...

These examples teach the model the **Protocol**. Without them, it might just output the answer directly or forget to wait for the Observation.

### Structured ReAct (JSON)

Parsing text (`Action: Search(...)`) is brittle.
Modern ReAct uses **Function Calling (JSON Mode)**.
*   **Thought:** Stored in `content`.
*   **Action:** Stored in `tool_calls`.
*   **Observation:** Stored in `tool` message.

This separates the reasoning (text) from the execution (structured data).

### Summary

*   **Stop Sequences:** Crucial. You must stop the model from generating the "Observation" itself.
*   **History Management:** You need to truncate the history if the trace gets too long (Sliding Window).

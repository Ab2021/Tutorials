# Lab 2: ReAct Agent from Scratch

## Objective
Build a **Reasoning + Acting (ReAct)** agent without using LangChain.
Understand the `Thought -> Action -> Observation` loop.

## 1. The Tools (`tools.py`)

```python
def calculator(expression):
    """Evaluates a math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error"

def search(query):
    """Simulates a search engine."""
    # Mock results
    if "capital of france" in query.lower():
        return "Paris is the capital of France."
    if "population of paris" in query.lower():
        return "The population of Paris is 2.1 million."
    return "No results found."

tools = {
    "calculator": calculator,
    "search": search
}
```

## 2. The Agent (`agent.py`)

```python
import re
from openai import OpenAI
from tools import tools

client = OpenAI()

SYSTEM_PROMPT = """
You are a ReAct agent.
You have access to the following tools:
- calculator: Evaluates math expressions.
- search: Searches for information.

Use the following format:
Question: the input question
Thought: you should always think about what to do
Action: the action to take, should be one of [calculator, search]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question
"""

def run_agent(question):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Question: {question}"}
    ]
    
    print(f"Question: {question}")
    
    for _ in range(5): # Max 5 steps
        # 1. Generate Thought + Action
        response = client.chat.completions.create(
            model="gpt-4",
            messages=messages,
            stop=["Observation:"] # Stop generation before hallucinating observation
        ).choices[0].message.content
        
        print(response)
        messages.append({"role": "assistant", "content": response})
        
        # 2. Parse Action
        if "Final Answer:" in response:
            return response.split("Final Answer:")[1].strip()
            
        action_regex = r"Action: (\w+)\nAction Input: (.*)"
        match = re.search(action_regex, response)
        
        if match:
            action_name = match.group(1)
            action_input = match.group(2)
            
            # 3. Execute Tool
            if action_name in tools:
                observation = tools[action_name](action_input)
                print(f"Observation: {observation}")
                
                # 4. Append Observation
                messages.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                print(f"Error: Tool {action_name} not found")
        else:
            print("Error: Could not parse action")
            break

# Run
q = "What is the population of the capital of France multiplied by 2?"
run_agent(q)
```

## 3. Analysis
Trace the output.
1.  Thought: I need to find the capital of France.
2.  Action: search("capital of france")
3.  Observation: Paris.
4.  Thought: Now I need the population of Paris.
5.  Action: search("population of paris")
6.  Observation: 2.1 million.
7.  Thought: Now I need to multiply 2.1 by 2.
8.  Action: calculator("2.1 * 2")
9.  Observation: 4.2.
10. Final Answer: 4.2 million.

## 4. Submission
Submit the console log of the agent solving the question.

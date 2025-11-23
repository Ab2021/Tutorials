# Day 56: Case Study: Building a Data Analyst Agent
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementation: The Pandas Agent

We will build a simplified version of the LangChain Pandas DataFrame Agent.

### 1. The Python REPL Tool

First, a tool that can execute Python code and return the output (stdout) + any created files.

```python
import sys
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

class PythonREPL:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.locals = {"df": df, "plt": plt, "pd": pd}
        
    def run(self, code: str):
        # Capture stdout
        old_stdout = sys.stdout
        sys.stdout = mystdout = StringIO()
        
        try:
            # Exec is dangerous! Use Docker in prod.
            exec(code, {}, self.locals)
            sys.stdout = old_stdout
            return mystdout.getvalue()
        except Exception as e:
            sys.stdout = old_stdout
            return f"Error: {str(e)}"

# Usage
# repl = PythonREPL(pd.read_csv("data.csv"))
# output = repl.run("print(df.describe())")
```

### 2. The Agent Prompt

We need to teach the agent how to use the `df` variable.

```python
SYSTEM_PROMPT = """
You are a Data Analyst. You have access to a pandas dataframe named `df`.
The first 5 rows are:
{df_head}

To answer questions, write Python code.
ALWAYS print the final answer.
If you generate a plot, save it to 'plot.png'.
"""

def create_analyst_agent(df):
    head = df.head().to_markdown()
    prompt = SYSTEM_PROMPT.format(df_head=head)
    
    # Initialize ReAct Agent with PythonREPL tool
    # ... (Standard LangChain setup)
    return agent
```

### 3. Handling Charts

When the agent generates a chart, we need to detect it.

```python
def run_and_check_plot(repl, code):
    output = repl.run(code)
    
    # Check if plot.png exists and was modified
    import os
    if os.path.exists("plot.png"):
        return f"{output}\n[System: Plot saved to plot.png]"
    return output
```

### 4. The Full Workflow

```python
def analyze_csv(file_path, query):
    df = pd.read_csv(file_path)
    repl = PythonREPL(df)
    
    # 1. Inspect Data
    print("Schema:", df.columns)
    
    # 2. Run Agent Loop
    # (Simplified manual loop for demonstration)
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(df_head=df.head())}]
    messages.append({"role": "user", "content": query})
    
    response = llm.chat(messages)
    
    if "```python" in response:
        # Extract code
        code = extract_code(response)
        print(f"Executing: {code}")
        
        # Run
        result = run_and_check_plot(repl, code)
        print(f"Result: {result}")
        
        # Feed back
        messages.append({"role": "assistant", "content": response})
        messages.append({"role": "user", "content": f"Execution Result: {result}"})
        
        # Final Answer
        final = llm.chat(messages)
        return final
```

### 5. Safety Improvements

In production, `exec()` is a non-starter.
**E2B (Code Interpreter SDK):**
```python
from e2b import Sandbox

def run_in_e2b(code):
    sandbox = Sandbox(template="pandas")
    # Upload CSV
    sandbox.upload_file("data.csv")
    # Run
    output = sandbox.run_code(code)
    return output.stdout
```

### Summary

The Data Analyst Agent combines:
1.  **State:** The Dataframe.
2.  **Tool:** The Code Executor.
3.  **Prompting:** Injecting the schema.
This pattern allows LLMs to solve math and data problems that are otherwise impossible for them.

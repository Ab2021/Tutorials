# Day 93: Data Analysis Agents
## Deep Dive - Internal Mechanics & Advanced Reasoning

### Implementing a Pandas Agent

We will build a simple agent that analyzes a CSV using Python `exec()`.

```python
import pandas as pd
import io

class PandasAgent:
    def __init__(self, client, df):
        self.client = client
        self.df = df
        self.history = []

    def ask(self, question):
        # 1. Inspect Data
        buffer = io.StringIO()
        self.df.info(buf=buffer)
        schema = buffer.getvalue()
        head = self.df.head(3).to_markdown()
        
        # 2. Generate Code
        prompt = f"""
        You have a pandas dataframe named 'df'.
        Schema:
        {schema}
        Head:
        {head}
        
        Question: {question}
        Write python code to answer this. Assign the result to a variable 'result'.
        Do not use print().
        """
        
        code = self.client.chat.completions.create(
            model="gpt-4-turbo",
            messages=[{"role": "user", "content": prompt}]
        ).choices[0].message.content
        
        # Clean code (remove markdown backticks)
        code = code.replace("```python", "").replace("```", "").strip()
        
        # 3. Execute (Unsafe - Demo only)
        local_scope = {"df": self.df}
        try:
            exec(code, {}, local_scope)
            return local_scope.get("result", "No result variable found.")
        except Exception as e:
            return f"Error: {str(e)}"

# Usage
df = pd.read_csv("sales.csv")
agent = PandasAgent(client, df)
print(agent.ask("What is the total revenue?"))
```

### Advanced: Text-to-SQL with Correction

```python
def run_sql_agent(query, schema):
    for attempt in range(3):
        sql = generate_sql(query, schema, error_log if attempt > 0 else "")
        try:
            return execute_query(sql)
        except Exception as e:
            error_log = str(e)
            print(f"SQL Failed: {error_log}. Retrying...")
    return "Failed"
```

### Visualizing Data

To handle plots:
1.  Agent writes code: `plt.savefig('plot.png')`.
2.  Sandbox captures the file.
3.  UI displays the image.

### Summary

*   **Context is King:** The model needs to know the column names and types.
*   **Iterative Repair:** The model *will* write bad SQL/Python. The loop (Try -> Catch -> Retry) is essential.
*   **Sandboxing:** Never use `exec()` in production without a container.

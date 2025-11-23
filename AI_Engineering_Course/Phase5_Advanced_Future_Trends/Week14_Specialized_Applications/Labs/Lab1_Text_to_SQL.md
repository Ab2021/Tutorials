# Lab 1: Text-to-SQL Agent

## Objective
Convert English to SQL safely.
We will use `sqlite3` and OpenAI.

## 1. The Database

```python
import sqlite3
conn = sqlite3.connect("sales.db")
conn.execute("CREATE TABLE sales (id INT, amount INT, region TEXT)")
conn.execute("INSERT INTO sales VALUES (1, 100, 'US'), (2, 200, 'EU')")
```

## 2. The Agent (`sql_agent.py`)

```python
from openai import OpenAI
client = OpenAI()

SCHEMA = "Table sales: id (INT), amount (INT), region (TEXT)"

def ask_database(question):
    # 1. Generate SQL
    prompt = f"""
    Schema: {SCHEMA}
    Question: {question}
    Write a SQL query. Output ONLY the SQL.
    """
    sql = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    ).choices[0].message.content
    
    print(f"Generated SQL: {sql}")
    
    # 2. Execute (Read-Only recommended)
    try:
        cursor = conn.execute(sql)
        return cursor.fetchall()
    except Exception as e:
        return f"Error: {e}"

print(ask_database("What is the total sales amount?"))
```

## 3. Challenge
*   **Correction Loop:** If execution fails, feed the error back to the LLM to fix the SQL.

## 4. Submission
Submit the generated SQL and the result.

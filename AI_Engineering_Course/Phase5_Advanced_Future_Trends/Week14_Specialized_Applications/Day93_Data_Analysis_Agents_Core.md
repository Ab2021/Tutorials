# Day 93: Data Analysis Agents
## Core Concepts & Theory

### The Analyst Agent

LLMs are great at writing code (Day 92).
Data Analysis Agents use this ability to **execute** code to answer questions about data.
*   *User:* "What is the average sales per region?"
*   *Agent:* Writes SQL or Python -> Executes -> Reads Result -> Explains.

### 1. Text-to-SQL

The standard pattern for structured data.
*   **Schema Linking:** The hardest part. Mapping "sales" (User) to `t_transactions.amt_usd` (DB).
*   **Dialect:** Postgres vs Snowflake vs BigQuery.
*   **Correction:** If SQL fails, the agent reads the error and retries.

### 2. The Code Interpreter Pattern (Pandas)

For unstructured or complex analysis (e.g., "Plot a trend line").
*   **Environment:** A sandboxed Python REPL.
*   **Workflow:**
    1.  Load CSV into `df`.
    2.  `df.head()` to understand structure.
    3.  Write analysis code.
    4.  Capture `stdout` and plots.

### 3. RAG for Data (Text-to-BI)

Injecting schema info into the context.
*   **DDL Injection:** `CREATE TABLE ...`
*   **Sample Rows:** Providing 3 rows helps the model understand data format (e.g., date formats).
*   **Documentation:** Providing a "Data Dictionary" (Column descriptions).

### 4. Security (The Sandbox)

You cannot let an LLM run `os.system('rm -rf /')`.
*   **Docker:** Run every session in an ephemeral container.
*   **WASM:** Run Python in the browser (Pyodide) or server-side WASM.
*   **Read-Only:** Database credentials should be Read-Only.

### Summary

Data Agents democratize data access. They turn English into SQL/Python. The challenge is **Reliability** (getting the right number) and **Security** (not leaking data).

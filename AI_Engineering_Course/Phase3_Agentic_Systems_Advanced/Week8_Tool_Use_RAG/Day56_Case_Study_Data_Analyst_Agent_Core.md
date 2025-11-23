# Day 56: Case Study: Building a Data Analyst Agent
## Core Concepts & Theory

### The Goal

We will synthesize everything we've learned this week (Tool Use, RAG, Security, Eval) to build a **Data Analyst Agent**.
**Mission:** "Take a CSV file, analyze it, and generate a report with charts."

### 1. Architecture

*   **Input:** User uploads a CSV (e.g., `sales_data.csv`).
*   **Tools:**
    *   `read_csv`: Pandas dataframe loader.
    *   `python_repl`: To execute analysis code (matplotlib/seaborn).
    *   `report_writer`: To save the final insights.
*   **Memory:** Needs to remember the column names and previous analysis steps.
*   **Security:** Sandboxed execution for the Python REPL.

### 2. The "Code Interpreter" Pattern

This is the most powerful pattern for data analysis. Instead of asking the LLM to *simulate* math (which it's bad at), we ask it to *write code* to do math (which it's good at).
*   *User:* "What is the correlation between price and sales?"
*   *Agent:* Writes `df['price'].corr(df['sales'])`.
*   *System:* Executes code. Returns `0.85`.
*   *Agent:* "There is a strong positive correlation of 0.85."

### 3. Handling Large Data

We cannot put the whole CSV in the context window.
**Strategy:**
1.  **Head:** Load only the first 5 rows (`df.head()`) into the context. This gives the schema.
2.  **Stats:** Load `df.describe()` to give distribution stats.
3.  **Code:** The agent writes code that operates on the *full* dataframe (loaded in the sandbox), but only sees the *summary* in the chat.

### 4. Iterative Refinement (ReAct)

Data analysis is exploratory.
*   *Thought 1:* "I need to check for missing values."
*   *Action 1:* `df.isnull().sum()`
*   *Observation 1:* "Column 'age' has 50 missing values."
*   *Thought 2:* "I should fill them with the median."
*   *Action 2:* `df['age'].fillna(df['age'].median(), inplace=True)`

### 5. Visualization

The agent needs to generate images.
*   *Code:* `plt.savefig('plot.png')`
*   *System:* Detects the file creation and displays it to the user.

### 6. Evaluation

How do we test this?
*   **Golden Dataset:** A set of CSVs with known answers ("What is the max sales in Q3?").
*   **Execution Check:** Did the code run without errors?
*   **Visual Check:** Did it generate a valid PNG?

### Summary

The Data Analyst Agent is the "Hello World" of advanced agentic systems. It demonstrates the power of **Tool Use** (Python REPL), **Context Management** (Dataframes), and **Multi-step Reasoning**.

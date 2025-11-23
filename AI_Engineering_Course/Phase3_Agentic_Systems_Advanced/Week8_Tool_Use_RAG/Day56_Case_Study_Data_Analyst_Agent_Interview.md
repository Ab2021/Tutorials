# Day 56: Case Study: Building a Data Analyst Agent
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: Why use a "Code Interpreter" instead of just passing the CSV text to the LLM?

**Answer:**
*   **Context Limit:** CSVs can be GBs in size. LLMs have limited context (128k tokens). You can't fit the whole file.
*   **Hallucination:** LLMs are bad at arithmetic. If you ask "What is the average of this column?", it might guess. Python `df.mean()` is exact.
*   **Complexity:** Code allows for complex logic (group by, filter, join) that is hard to express purely in text generation.

#### Q2: How do you handle "Dirty Data" with an agent?

**Answer:**
The agent needs to be robust.
*   **Error Recovery:** If `df['date']` fails because of mixed formats, the agent should see the error and try `pd.to_datetime(..., errors='coerce')`.
*   **Exploration:** The agent should start by running `df.info()` or `df.head()` to understand the data quality before trying to analyze it.

#### Q3: What is the risk of "Infinite Loops" in code generation?

**Answer:**
The agent might write `while True: pass`.
**Mitigation:**
*   **Timeouts:** The sandbox must kill the process after 5 seconds.
*   **Resource Limits:** Limit RAM and CPU usage of the container.

#### Q4: How do you persist state between code executions?

**Answer:**
*   **Stateful Session:** The Python REPL must be persistent. If Step 1 defines `x = 5`, Step 2 must be able to print `x`.
*   **Implementation:** Use a Jupyter Kernel or a persistent process with `code.InteractiveConsole`. Simple `exec()` calls are stateless unless you pass the `locals()` dictionary around.

### Production Challenges

#### Challenge 1: Library Dependencies

**Scenario:** User asks "Use the Prophet library to forecast sales." Prophet isn't installed in the container.
**Root Cause:** Static environment.
**Solution:**
*   **Pre-baked Images:** Include common data science libs (pandas, numpy, scipy, sklearn, statsmodels) in the Docker image.
*   **Dynamic Install:** Allow the agent to run `!pip install prophet` (Risky! Needs internet access and strict whitelisting).

#### Challenge 2: Plotting & UI Integration

**Scenario:** Agent generates a plot. How do you show it in the Chat UI?
**Root Cause:** Text-only interface.
**Solution:**
*   **Base64:** Have the Python script save the plot to a BytesIO buffer, base64 encode it, and print it to stdout. The UI decodes and renders it.
*   **File Storage:** Save to S3, return the URL.

#### Challenge 3: Explaining the "Why"

**Scenario:** Agent says "Sales dropped 50%." User asks "Why?".
**Root Cause:** The agent only calculated the number, it didn't investigate causality.
**Solution:**
*   **Reasoning Prompt:** Instruct the agent: "After calculating the number, explain *what it means* in the context of the data."
*   **Drill Down:** If sales dropped, the agent should autonomously check "Did price change?", "Did traffic drop?".

#### Challenge 4: Security (The "rm -rf" problem)

**Scenario:** User uploads a malicious CSV that exploits a vulnerability in pandas (unlikely but possible) or prompts the agent to delete files.
**Root Cause:** Unsafe execution.
**Solution:**
*   **Ephemeral Containers:** Every session gets a fresh Docker container. It is destroyed immediately after the session ends. Nothing persists.

### System Design Scenario: Enterprise Data Chatbot

**Requirement:** Chat with your Snowflake Data Warehouse.
**Design:**
1.  **Schema Fetcher:** A tool that queries `INFORMATION_SCHEMA` to get table definitions.
2.  **SQL Generator:** Agent writes SQL.
3.  **Validator:** A "Dry Run" step checks the SQL for syntax errors and forbidden tables.
4.  **Executor:** Runs the SQL.
5.  **Python Analyst:** Takes the SQL result (if < 1MB) and uses Python to plot it.
6.  **UI:** Renders Markdown + Images.

### Summary Checklist for Production
*   [ ] **Sandboxing:** Use **E2B** or **Firecracker** for code execution.
*   [ ] **State:** Ensure the REPL maintains variable state across turns.
*   [ ] **Libs:** Pre-install **Pandas/Matplotlib** in the image.
*   [ ] **Limits:** Enforce **Timeouts** and **Memory Limits**.
*   [ ] **Visuals:** Handle **Image Output** gracefully.

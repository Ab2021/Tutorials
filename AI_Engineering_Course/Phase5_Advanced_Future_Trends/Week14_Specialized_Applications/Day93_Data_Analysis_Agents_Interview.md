# Day 93: Data Analysis Agents
## Interview Questions & Production Challenges

### Interview Questions

#### Q1: How do you handle "Hallucinated Columns" in Text-to-SQL?

**Answer:**
The model guesses a column `user_name` but the DB has `username`.
*   **Schema Injection:** Always include the full DDL in the prompt.
*   **Vector Search:** If the schema is too big (1000 tables), use RAG to retrieve only relevant tables/columns based on the user query.

#### Q2: What is "Chain of Table" reasoning?

**Answer:**
A prompting strategy.
1.  **Select:** Choose relevant tables.
2.  **Filter:** Choose relevant rows.
3.  **Transform:** Apply math/aggregation.
It breaks the SQL generation into logical steps, improving accuracy on complex joins.

#### Q3: Why use Python (Pandas) instead of SQL?

**Answer:**
*   **Complexity:** SQL is bad at complex math (e.g., "Calculate the correlation coefficient"). Python is native for this.
*   **Visualization:** SQL cannot generate charts. Python (Matplotlib) can.
*   **Data Cleaning:** Python is better at handling messy dates/strings.

#### Q4: How do you secure a Code Interpreter?

**Answer:**
*   **Network Isolation:** No internet access.
*   **Resource Limits:** Max 512MB RAM, 10s CPU.
*   **Syscall Filtering:** Block `fork`, `exec`, `socket` (using seccomp).

### Production Challenges

#### Challenge 1: The "Ambiguous Question"

**Scenario:** User asks "Show me the top users."
**Root Cause:** "Top" is undefined. (By spend? By visits? By tenure?)
**Solution:**
*   **Clarification:** The agent should ask: "By 'top', do you mean highest revenue or most logins?"
*   **Assumptions:** "I assumed you meant Revenue. Here is the data."

#### Challenge 2: Large Result Sets

**Scenario:** Query returns 1M rows. The agent tries to read them all into context to "summarize" them. OOM.
**Root Cause:** Unbounded fetch.
**Solution:**
*   **Limit:** Always append `LIMIT 100` to generated SQL.
*   **Aggregation:** Force the agent to write `COUNT` or `SUM` queries, not `SELECT *`.

#### Challenge 3: Schema Drift

**Scenario:** DB schema changes. Agent breaks.
**Root Cause:** Cached schema in Vector DB.
**Solution:**
*   **Sync:** Hook into the DB migration pipeline to update the Vector DB whenever the schema changes.

### System Design Scenario: Self-Service BI Tool

**Requirement:** Allow non-technical PMs to query the Data Warehouse.
**Design:**
1.  **Semantic Layer:** Define "Metrics" (e.g., `Revenue = sum(orders.amt)`). Don't expose raw tables.
2.  **Router:**
    *   Simple lookup -> SQL.
    *   Trend analysis -> Python.
3.  **Caching:** Cache the SQL/Code for common questions ("What is daily active users?").
4.  **Feedback:** User thumbs up/down the result. Fine-tune the model on the correct SQL.

### Summary Checklist for Production
*   [ ] **Read-Only User:** The DB user must have `SELECT` only.
*   [ ] **Timeout:** Kill queries > 30s.
*   [ ] **Explanation:** Always explain *how* the answer was derived ("I filtered for status='active'").

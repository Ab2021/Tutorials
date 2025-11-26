# Day 7: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the "N+1 Select Problem"?
**Answer:**
*   **Definition**: It happens when code executes N additional queries to fetch related data for a list of N items.
*   **Example**:
    ```python
    users = db.query("SELECT * FROM users") # 1 Query
    for user in users:
        print(user.orders) # N Queries (SELECT * FROM orders WHERE user_id = ?)
    ```
*   **Fix**: Use a **Join** (`SELECT * FROM users JOIN orders ...`) or **Eager Loading** (fetching all IDs in one go: `SELECT * FROM orders WHERE user_id IN (...)`).

### Q2: Why shouldn't we index every column?
**Answer:**
1.  **Write Penalty**: Every `INSERT`, `UPDATE`, or `DELETE` requires updating the table *and* every index. Too many indexes slow down writes significantly.
2.  **Storage Cost**: Indexes take up disk space and RAM.
3.  **Optimizer Confusion**: Too many indexes might confuse the query planner, leading to sub-optimal paths.

### Q3: Explain "Dirty Read" vs "Phantom Read".
**Answer:**
*   **Dirty Read**: Reading uncommitted data. Transaction A writes X=5 (not committed). Transaction B reads X=5. Transaction A rolls back. B has bad data.
*   **Phantom Read**: A phenomenon where a query returns a different set of rows when executed twice in the same transaction. Transaction A reads "all users in NY" (5 users). Transaction B inserts a new user in NY. Transaction A reads again and sees 6 users.

---

## Scenario-Based Questions

### Q4: You have a query `SELECT * FROM logs WHERE message LIKE '%error%'` that is running very slow. How do you fix it?
**Answer:**
*   **Problem**: Leading wildcard (`%error%`) prevents using a standard B-Tree index. The DB must scan every row.
*   **Solution 1 (Postgres)**: Use a **Trigram Index** (`pg_trgm` extension) or a **GIN Index** for full-text search (`tsvector`).
*   **Solution 2 (External)**: Offload logs to an Elasticsearch/OpenSearch cluster which is optimized for text search (Inverted Index).

### Q5: A developer wants to store a JSON object in a Postgres text column. Is this a good idea?
**Answer:**
*   **It depends**.
*   **Bad**: If they store it as `TEXT` or `VARCHAR`, the DB treats it as a string. You can't query fields inside it efficiently.
*   **Good**: If they use the `JSONB` data type. Postgres parses the JSON, stores it in a binary format, and allows **indexing specific keys** (e.g., `CREATE INDEX ON table((data->>'id'))`). This gives NoSQL-like flexibility with SQL power.

---

## Behavioral / Role-Specific Questions

### Q6: How do you decide when to denormalize a database schema?
**Answer:**
*   **Default**: Start Normalized (3NF). It ensures data consistency and saves space.
*   **Trigger**: When performance monitoring shows that `JOIN`s are the bottleneck for a critical, high-frequency read path.
*   **Action**: I would duplicate the data (e.g., adding `author_name` to `books` table to avoid joining `authors`).
*   **Trade-off**: I must implement logic (triggers or app code) to update the denormalized field whenever the source changes (e.g., if Author changes name, update all their books).

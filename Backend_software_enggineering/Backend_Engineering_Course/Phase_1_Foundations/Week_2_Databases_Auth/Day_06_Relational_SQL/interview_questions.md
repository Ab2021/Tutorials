# Day 6: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between `WHERE` and `HAVING`?
**Answer:**
*   **WHERE**: Filters rows *before* the aggregation (GROUP BY) happens.
    *   `SELECT * FROM orders WHERE amount > 100`
*   **HAVING**: Filters groups *after* the aggregation happens.
    *   `SELECT user_id, SUM(amount) FROM orders GROUP BY user_id HAVING SUM(amount) > 1000`
*   *Mnemonic*: WHERE is for rows, HAVING is for groups.

### Q2: Explain the difference between `INNER JOIN` and `LEFT JOIN`.
**Answer:**
*   **INNER JOIN**: Returns only the rows where there is a match in **both** tables. If a user has no orders, they won't appear in the result.
*   **LEFT JOIN**: Returns **all** rows from the left table (e.g., Users), and the matching rows from the right table (Orders). If a user has no orders, the order columns will be `NULL`.

### Q3: What is a Foreign Key constraint and why is it useful?
**Answer:**
*   **Definition**: A rule that ensures the value in a column (e.g., `orders.user_id`) matches a value in the referenced table's Primary Key (`users.id`).
*   **Utility**: It enforces **Referential Integrity**. It prevents "orphaned records". You cannot delete a User if they still have Orders (unless `ON DELETE CASCADE` is set).

---

## Scenario-Based Questions

### Q4: You have a table `employees` with a `manager_id` column that references `id` in the same table. Write a query to find the names of employees and their managers.
**Answer:**
This requires a **Self Join**.
```sql
SELECT 
    e.name AS employee_name, 
    m.name AS manager_name
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```
*Note*: Use `LEFT JOIN` because the CEO has no manager (NULL `manager_id`). An `INNER JOIN` would exclude the CEO.

### Q5: A query `SELECT * FROM large_table` is crashing the application server. Why?
**Answer:**
*   **Issue**: The result set is too large to fit in the application's memory (RAM). The DB tries to send millions of rows over the network.
*   **Fixes**:
    1.  **Pagination**: Use `LIMIT` and `OFFSET` (or Cursor pagination).
    2.  **Streaming**: Use a database cursor (server-side cursor) to fetch rows in chunks.
    3.  **Projection**: Select only needed columns (`SELECT id, name`) instead of `*`.

---

## Behavioral / Role-Specific Questions

### Q6: How do you handle `NULL` values in SQL? Why are they dangerous?
**Answer:**
*   **Danger**: `NULL` means "unknown". It is not equal to zero or empty string.
    *   `NULL = NULL` is **False** (or NULL). You must use `IS NULL`.
    *   `COUNT(column)` ignores NULLs, but `COUNT(*)` counts them.
    *   `1 + NULL` = `NULL`.
*   **Handling**:
    *   Use `COALESCE(column, default_value)` to convert NULLs to a safe value.
    *   Enforce `NOT NULL` constraints in the schema wherever possible.

# Day 7: Normalization, Indexes & Transactions

## 1. Database Normalization

Normalization is the process of organizing data to reduce redundancy and improve integrity.

### 1.1 The Normal Forms
*   **1NF (First Normal Form)**: Atomicity.
    *   Rule: No repeating groups or arrays. Each cell must contain a single value.
    *   *Bad*: `tags: "red, blue, green"`
    *   *Good*: A separate `product_tags` table.
*   **2NF (Second Normal Form)**: No Partial Dependencies.
    *   Rule: Must be in 1NF, and all non-key columns must depend on the *entire* Primary Key.
    *   *Bad*: Table `OrderItems (order_id, product_id, product_name)`. `product_name` depends only on `product_id`, not the full key.
    *   *Good*: Move `product_name` to `Products` table.
*   **3NF (Third Normal Form)**: No Transitive Dependencies.
    *   Rule: Must be in 2NF, and non-key columns should not depend on other non-key columns.
    *   *Bad*: Table `Users (id, zip_code, city)`. `city` depends on `zip_code`.
    *   *Good*: Move `zip_code` and `city` to a `Locations` table.

### 1.2 When to Denormalize?
Normalization optimizes for **Writes** (no duplicate updates). Denormalization optimizes for **Reads** (fewer joins).
*   *Scenario*: An analytics dashboard needs to show "Total Sales by City".
*   *Normalized*: Join Users -> Orders -> OrderItems. Slow.
*   *Denormalized*: Add `city` column to `Orders` table. Fast read, but redundant storage.

---

## 2. Indexes: The Speed Layer

An index is a data structure that improves the speed of data retrieval operations on a database table at the cost of additional writes and storage space.

### 2.1 B-Tree Index (The Default)
*   **Structure**: A balanced tree.
*   **Operations**: `O(log N)` for search, insert, delete.
*   **Use Case**: Equality (`=`), Range (`<`, `>`, `BETWEEN`), Sorting (`ORDER BY`).
*   *Example*: `CREATE INDEX idx_users_email ON users(email);`

### 2.2 Hash Index
*   **Structure**: Hash Table.
*   **Operations**: `O(1)` for equality.
*   **Limitation**: Cannot handle ranges (`> 5`). Postgres rarely uses these as B-Trees are almost as fast and more versatile.

### 2.3 Composite Index (Multi-column)
*   **Rule**: Left-to-Right matching.
*   *Index*: `(lastname, firstname)`
*   *Query*: `WHERE lastname = 'Smith'` -> **Uses Index**.
*   *Query*: `WHERE firstname = 'John'` -> **Scans Table** (Index useless).
*   *Query*: `WHERE lastname = 'Smith' AND firstname = 'John'` -> **Uses Index**.

### 2.4 Covering Index
An index that contains *all* the fields required by the query, so the DB doesn't need to look up the actual row (Heap).
*   *Query*: `SELECT email FROM users WHERE id = 5;`
*   *Index*: `(id, email)` -> The DB gets the answer directly from the index.

---

## 3. Transactions & Isolation Levels

We know ACID. Let's talk about the "I" (Isolation). How much do concurrent transactions see of each other?

### 3.1 The Levels (ANSI SQL Standard)
1.  **Read Uncommitted**: You can see data written by other transactions that haven't committed yet.
    *   *Risk*: Dirty Reads. (They rollback, you acted on phantom data).
    *   *Postgres*: Maps this to Read Committed.
2.  **Read Committed** (Default in Postgres): You only see data that has been committed.
    *   *Risk*: Non-repeatable Reads. (You read row A, someone updates it, you read row A again and it changed).
3.  **Repeatable Read**: You see a snapshot of the DB as it was when your transaction started.
    *   *Risk*: Phantom Reads (New rows inserted by others might appear in range queries, though Postgres prevents this too).
4.  **Serializable**: Strict serial execution.
    *   *Cost*: High. DB will throw errors ("Serialization Failure") if it detects conflict, forcing you to retry.

---

## 4. Query Optimization Basics

How to fix a slow query?

### 4.1 EXPLAIN ANALYZE
This command runs the query and shows the execution plan.
```sql
EXPLAIN ANALYZE SELECT * FROM users WHERE email = 'bob@example.com';
```
*   **Seq Scan**: The DB read every row. Bad for large tables.
*   **Index Scan**: The DB used the B-Tree. Good.
*   **Index Only Scan**: The DB used a covering index. Best.

### 4.2 Common Pitfalls
1.  **Functions on Columns**: `WHERE YEAR(created_at) = 2023`. This disables the index on `created_at`.
    *   *Fix*: `WHERE created_at >= '2023-01-01' AND created_at < '2024-01-01'`.
2.  **Leading Wildcards**: `LIKE '%gmail.com'`. Cannot use B-Tree.
    *   *Fix*: Reverse string index or Trigram index.
3.  **OR conditions**: `WHERE id = 5 OR email = 'bob'`. Might force a scan.
    *   *Fix*: `UNION ALL` of two queries.

---

## 5. Summary

Today we refined our database skills.
*   **Normalization**: Organize data to prevent anomalies.
*   **Indexes**: B-Trees are your best friend for performance.
*   **Isolation**: Trade-off between consistency and concurrency.

**Tomorrow (Day 8)**: We break the rules. We will look at **NoSQL** (MongoDB and Redis) and see when to throw away schemas and joins for raw speed and scale.

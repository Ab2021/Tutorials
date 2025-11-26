# Day 6: Relational Databases & SQL Basics

## 1. The Relational Model

Despite the hype around NoSQL, **Relational Databases (RDBMS)** remain the backbone of 90% of backend systems. Understanding them deeply is non-negotiable.

### 1.1 Core Concepts
*   **Table (Relation)**: A set of tuples (rows) sharing the same attributes (columns).
*   **Row (Tuple)**: A single record.
*   **Column (Attribute)**: A specific piece of data (e.g., `email`, `age`) with a defined type.
*   **Schema**: The blueprint defining tables, columns, and constraints.

### 1.2 Constraints (The Guardrails)
*   **Primary Key (PK)**: Unique identifier for a row. (e.g., `id`, `uuid`). *Cannot be NULL*.
*   **Foreign Key (FK)**: A reference to a PK in another table. Enforces referential integrity. (e.g., `user_id` in `orders` table).
*   **Unique**: Ensures no duplicates in a column (e.g., `email`).
*   **Not Null**: Mandatory field.
*   **Check**: Custom validation (e.g., `age > 18`).

---

## 2. SQL: The Language of Data

SQL (Structured Query Language) is declarative. You tell the DB *what* you want, not *how* to get it.

### 2.1 DDL (Data Definition Language)
Defining the structure.
```sql
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id), -- Foreign Key
    total_amount DECIMAL(10, 2) CHECK (total_amount >= 0),
    status VARCHAR(20) DEFAULT 'pending'
);
```

### 2.2 DML (Data Manipulation Language)
Changing the data.

*   **INSERT**:
    ```sql
    INSERT INTO users (username, email) VALUES ('alice', 'alice@example.com');
    ```
*   **UPDATE**:
    ```sql
    UPDATE orders SET status = 'shipped' WHERE id = 101;
    ```
*   **DELETE**:
    ```sql
    DELETE FROM users WHERE id = 5; -- Dangerous! Always use WHERE.
    ```

### 2.3 DQL (Data Query Language)
Reading the data.

*   **SELECT**:
    ```sql
    SELECT username, email FROM users WHERE created_at > '2025-01-01';
    ```
*   **ORDER BY**:
    ```sql
    SELECT * FROM orders ORDER BY total_amount DESC;
    ```
*   **LIMIT / OFFSET** (Pagination):
    ```sql
    SELECT * FROM users LIMIT 10 OFFSET 20; -- Page 3
    ```

---

## 3. Deep Dive: How Postgres Stores Data

To be an advanced engineer, you must look under the hood.

### 3.1 The Heap and The Page
*   **Page**: Postgres stores data in fixed-size blocks called "Pages" (usually 8KB).
*   **Heap**: The collection of pages containing the actual table data.
*   **Tuple**: A row version. Postgres uses MVCC (Multi-Version Concurrency Control), so an update actually creates a *new* tuple and marks the old one as dead.

### 3.2 TOAST (The Oversized Attribute Storage Technique)
*   If a row is too big for a page (e.g., a massive JSON blob or text), Postgres slices it up and stores it in a separate "TOAST table".
*   *Performance Tip*: Selecting columns you don't need (like a huge `bio` text field) forces the DB to fetch from TOAST, slowing down the query. **Avoid `SELECT *`**.

---

## 4. Joins: The Power of Relations

Joins allow you to combine data from multiple tables.

### 4.1 Types of Joins
1.  **INNER JOIN**: Returns rows when there is a match in *both* tables.
    ```sql
    SELECT u.username, o.total_amount
    FROM users u
    INNER JOIN orders o ON u.id = o.user_id;
    ```
2.  **LEFT (OUTER) JOIN**: Returns all rows from the left table, and the matched rows from the right table. If no match, NULL.
    *   *Use Case*: "Show me all users, and their orders if they have any."
3.  **RIGHT JOIN**: Opposite of Left Join. Rarely used.
4.  **FULL OUTER JOIN**: Returns rows when there is a match in *one* of the tables.

### 4.2 Aggregation
*   **GROUP BY**: Group rows that have the same values.
*   **HAVING**: Filter groups (like `WHERE` but for groups).
    ```sql
    SELECT user_id, COUNT(*) as order_count
    FROM orders
    GROUP BY user_id
    HAVING COUNT(*) > 5; -- Only VIP users
    ```

---

## 5. Summary

Today we covered the bedrock of data storage.
*   **Schema Design**: Defining types and constraints.
*   **SQL**: The syntax for CRUD operations.
*   **Internals**: Pages, Heaps, and TOAST.
*   **Joins**: Combining data.

**Tomorrow (Day 7)**: We will learn how to design *good* schemas (Normalization) and how to make queries fast (Indexes).

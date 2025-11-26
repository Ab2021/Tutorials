# Day 6: Relational Databases & Advanced SQL - Mastering the Foundation

## Table of Contents
1. [SQL Fundamentals Recap](#1-sql-fundamentals-recap)
2. [Advanced Query Techniques](#2-advanced-query-techniques)
3. [Joins Mastery](#3-joins-mastery)
4. [Subqueries & CTEs](#4-subqueries--ctes)
5. [Window Functions](#5-window-functions)
6. [Transactions & Isolation Levels](#6-transactions--isolation-levels)
7. [Query Optimization](#7-query-optimization)
8. [Performance Tuning](#8-performance-tuning)
9. [Best Practices](#9-best-practices)
10. [Summary](#10-summary)

---

## 1. SQL Fundamentals Recap

### 1.1 SQL Statement Types

**DDL (Data Definition Language)**:
```sql
CREATE TABLE users (id SERIAL PRIMARY KEY, name VARCHAR(100));
ALTER TABLE users ADD COLUMN email VARCHAR(255);
DROP TABLE old_table;
```

**DML (Data Manipulation Language)**:
```sql
INSERT INTO users (name, email) VALUES ('Alice', 'alice@example.com');
UPDATE users SET email = 'newemail@example.com' WHERE id = 1;
DELETE FROM users WHERE id = 1;
```

**DQL (Data Query Language)**:
```sql
SELECT * FROM users WHERE created_at > '2024-01-01';
```

**DCL (Data Control Language)**:
```sql
GRANT SELECT ON users TO read_only_user;
REVOKE INSERT ON users FROM junior_dev;
```

### 1.2 Basic SELECT Structure

```sql
SELECT 
    column1, 
    column2,
    aggregate_function(column3)
FROM 
    table_name
JOIN 
    other_table ON condition
WHERE 
    filter_conditions
GROUP BY 
    column1, column2
HAVING 
    aggregate_filter
ORDER BY 
    column1 DESC
LIMIT 10 OFFSET 20;
```

**Execution Order** (different from writing order):
1. FROM + JOINs
2. WHERE
3. GROUP BY
4. HAVING
5. SELECT
6. ORDER BY
7. LIMIT/OFFSET

---

## 2. Advanced Query Techniques

### 2.1 CASE Expressions

**Simple CASE**:
```sql
SELECT 
    name,
    CASE status
        WHEN 'active' THEN 'Active User'
        WHEN 'pending' THEN 'Pending Approval'
        ELSE 'Unknown'
    END AS status_label
FROM users;
```

**Searched CASE**:
```sql
SELECT 
    name,
    CASE 
        WHEN age < 18 THEN 'Minor'
        WHEN age BETWEEN 18 AND 64 THEN 'Adult'
        ELSE 'Senior'
    END AS age_group
FROM users;
```

**Pivot with CASE**:
```sql
SELECT 
    user_id,
    SUM(CASE WHEN status = 'completed' THEN 1 ELSE 0 END) AS completed_orders,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) AS pending_orders,
    SUM(CASE WHEN status = 'canceled' THEN 1 ELSE 0 END) AS canceled_orders
FROM orders
GROUP BY user_id;
```

### 2.2 COALESCE & NULLIF

**COALESCE** - Return first non-null:
```sql
SELECT 
    name,
    COALESCE(phone, email, 'No contact') AS contact_method
FROM users;
```

**NULLIF** - Return NULL if values equal:
```sql
SELECT 
    name,
    NULLIF(discount, 0) AS discount  -- NULL if discount is 0
FROM products;
```

### 2.3 String Functions

```sql
-- Concatenation
SELECT first_name || ' ' || last_name AS full_name FROM users;

-- Pattern matching
SELECT * FROM users WHERE email LIKE '%@gmail.com';
SELECT * FROM users WHERE email ILIKE '%GMAIL%';  -- Case-insensitive (Postgres)

-- Regular expressions (Postgres)
SELECT * FROM users WHERE email ~ '^[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}$';

-- String manipulation
SELECT 
    UPPER(name),
    LOWER(email),
    LENGTH(bio),
    SUBSTRING(phone FROM 1 FOR 3) AS area_code,
    TRIM(name)
FROM users;
```

###  2.4 Date/Time Functions

```sql
-- Current timestamp
SELECT NOW(), CURRENT_DATE, CURRENT_TIME;

-- Date arithmetic
SELECT created_at + INTERVAL '7 days' AS one_week_later FROM orders;
SELECT AGE(NOW(), birth_date) AS age FROM users;

-- Extract parts
SELECT 
    EXTRACT(YEAR FROM created_at) AS year,
    EXTRACT(MONTH FROM created_at) AS month,
    EXTRACT(DAY FROM created_at) AS day
FROM orders;

-- Date truncation
SELECT DATE_TRUNC('month', created_at) AS month_start FROM orders;
```

---

## 3. Joins Mastery

### 3.1 Join Types Visualized

```
Table A (users):         Table B (orders):
id | name                 id | user_id | amount
1  | Alice                1  | 1       | 100
2  | Bob                  2  | 1       | 200
3  | Charlie              3  | 3       | 50
```

#### INNER JOIN
```sql
SELECT  users.name, orders.amount
FROM users
INNER JOIN orders ON users.id = orders.user_id;

Result:
Alice  | 100
Alice  | 200
Charlie| 50
-- Bob excluded (no orders)
```

#### LEFT JOIN (LEFT OUTER JOIN)
```sql
SELECT users.name, orders.amount
FROM users
LEFT JOIN orders ON users.id = orders.user_id;

Result:
Alice  | 100
Alice  | 200
Bob    | NULL  -- Included even without orders
Charlie| 50
```

#### RIGHT JOIN
```sql
SELECT users.name, orders.amount
FROM users
RIGHT JOIN orders ON users.id = orders.user_id;

-- Same as LEFT JOIN with tables swapped
```

#### FULL OUTER JOIN
```sql
SELECT users.name, orders.amount
FROM users
FULL OUTER JOIN orders ON users.id = orders.user_id;

-- Includes users without orders AND orders without users (if any)
```

#### CROSS JOIN (Cartesian Product)
```sql
SELECT users.name, products.name
FROM users
CROSS JOIN products;

-- Every user paired with every product (3 × 100 = 300 rows if 3 users, 100 products)
```

### 3.2 Self Joins

**Finding employees and their managers**:
```sql
CREATE TABLE employees (
    id INT PRIMARY KEY,
    name VARCHAR(100),
    manager_id INT
);

SELECT 
    e.name AS employee,
    m.name AS manager
FROM employees e
LEFT JOIN employees m ON e.manager_id = m.id;
```

### 3.3 Multi-Table Joins

```sql
SELECT 
    users.name,
    orders.id AS order_id,
    products.name AS product_name,
    order_items.quantity,
    order_items.price
FROM users
JOIN orders ON users.id = orders.user_id
JOIN order_items ON orders.id = order_items.order_id
JOIN products ON order_items.product_id = products.id
WHERE orders.status = 'completed';
```

---

## 4. Subqueries & CTEs

### 4.1 Subqueries in WHERE

```sql
-- Find users who have placed orders
SELECT * FROM users
WHERE id IN (SELECT DISTINCT user_id FROM orders);

-- Find users with above-average order totals
SELECT * FROM users
WHERE id IN (
    SELECT user_id 
    FROM orders 
    GROUP BY user_id
    HAVING SUM(amount) > (SELECT AVG(total) FROM (SELECT SUM(amount) AS total FROM orders GROUP BY user_id) subq)
);
```

### 4.2 Correlated Subqueries

```sql
-- Find users with their latest order date
SELECT 
    u.name,
    (SELECT MAX(created_at) FROM orders o WHERE o.user_id = u.id) AS last_order_date
FROM users u;

-- Find products with above-category-average price
SELECT *
FROM products p1
WHERE price > (
    SELECT AVG(price)
    FROM products p2
    WHERE p2.category_id = p1.category_id
);
```

### 4.3 Common Table Expressions (CTEs)

**Basic CTE**:
```sql
WITH monthly_sales AS (
    SELECT 
        DATE_TRUNC('month', created_at) AS month,
        SUM(amount) AS total_sales
    FROM orders
    GROUP BY month
)
SELECT * FROM monthly_sales WHERE total_sales > 10000;
```

**Multiple CTEs**:
```sql
WITH 
    active_users AS (
        SELECT id, name FROM users WHERE status = 'active'
    ),
    recent_orders AS (
        SELECT user_id, amount FROM orders WHERE created_at > NOW() - INTERVAL '30 days'
    )
SELECT 
    u.name,
    SUM(o.amount) AS total_spent
FROM active_users u
JOIN recent_orders o ON u.id = o.user_id
GROUP BY u.id, u.name;
```

**Recursive CTE** (Organizational hierarchy):
```sql
WITH RECURSIVE org_chart AS (
    -- Base case: top-level managers
    SELECT id, name, manager_id, 1 AS level
    FROM employees
    WHERE manager_id IS NULL
    
    UNION ALL
    
    -- Recursive case: employees reporting to previous level
    SELECT e.id, e.name, e.manager_id, oc.level + 1
    FROM employees e
    JOIN org_chart oc ON e.manager_id = oc.id
)
SELECT * FROM org_chart ORDER BY level, name;
```

---

## 5. Window Functions

### 5.1 Basic Window Functions

**ROW_NUMBER** - Sequential numbering:
```sql
SELECT 
    name,
    department,
    salary,
    ROW_NUMBER() OVER (PARTITION BY department ORDER BY salary DESC) AS rank_in_dept
FROM employees;

Result:
name    | department | salary | rank_in_dept
Alice   | Engineering| 120000 | 1
Bob     | Engineering| 110000 | 2
Charlie | Sales      | 90000  | 1
```

**RANK & DENSE_RANK**:
```sql
SELECT 
    name,
    score,
    RANK() OVER (ORDER BY score DESC) AS rank,
    DENSE_RANK() OVER (ORDER BY score DESC) AS dense_rank
FROM students;

Result (with ties):
name  | score | rank | dense_rank
Alice | 95    | 1    | 1
Bob   | 95    | 1    | 1
Carol | 90    | 3    | 2  -- RANK skips 2, DENSE_RANK doesn't
```

### 5.2 Aggregate Window Functions

```sql
SELECT 
    date,
    revenue,
    SUM(revenue) OVER (ORDER BY date ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative_revenue,
    AVG(revenue) OVER (ORDER BY date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS rolling_7day_avg
FROM daily_sales;
```

### 5.3 LAG & LEAD

```sql
SELECT 
    date,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY date) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY date) AS next_day_revenue,
    revenue - LAG(revenue, 1) OVER (ORDER BY date) AS day_over_day_change
FROM daily_sales;
```

### 5.4 NTILE (Buckets)

```sql
SELECT 
    user_id,
    total_spent,
    NTILE(4) OVER (ORDER BY total_spent DESC) AS quartile
FROM (
    SELECT user_id, SUM(amount) AS total_spent
    FROM orders
    GROUP BY user_id
) subq;
-- Quartile 1 = top 25% spenders
```

---

## 6. Transactions & Isolation Levels

### 6.1 ACID Transactions

```sql
BEGIN;  -- Start transaction

UPDATE accounts SET balance = balance - 100 WHERE id = 1;
UPDATE accounts SET balance = balance + 100 WHERE id = 2;

-- If any error, all changes rollback
COMMIT;  -- Make changes permanent
```

**Error handling (Postgres)**:
```sql
BEGIN;
    UPDATE accounts SET balance = balance - 100 WHERE id = 1;
    -- Imagine error occurs here
    UPDATE invalid_table SET foo = 'bar';  -- ERROR!
ROLLBACK;  -- Undo all changes
```

### 6.2 Savepoints

```sql
BEGIN;
    UPDATE users SET status = 'active' WHERE id = 1;
    
    SAVEPOINT sp1;
    
    UPDATE users SET status = 'banned' WHERE id = 2;
    -- Oops, wrong user
    
    ROLLBACK TO SAVEPOINT sp1;  -- Undo only changes after sp1
    
    UPDATE users SET status = 'banned' WHERE id = 3;  -- Correct user
COMMIT;
```

### 6.3 Isolation Levels

```sql
SET TRANSACTION ISOLATION LEVEL READ COMMITTED;
```

| Isolation Level | Dirty Read | Non-repeatable Read | Phantom Read |
|:----------------|:-----------|:--------------------|:-------------|
| READ UNCOMMITTED | ✅ Possible | ✅ Possible | ✅ Possible |
| READ COMMITTED (default) | ❌ Prevented | ✅ Possible | ✅ Possible |
| REPEATABLE READ | ❌ Prevented | ❌ Prevented | ✅ Possible |
| SERIALIZABLE | ❌ Prevented | ❌ Prevented | ❌ Prevented|

**Dirty Read Example**:
```sql
-- Transaction 1
BEGIN;
UPDATE accounts SET balance = 1000 WHERE id = 1;
-- Not committed yet

-- Transaction 2 (READ UNCOMMITTED)
SELECT balance FROM accounts WHERE id = 1;  → 1000 (dirty read!)

-- Transaction 1
ROLLBACK;  -- Transaction 2 saw data that never existed!
```

**Non-repeatable Read Example**:
```sql
-- Transaction 1
BEGIN;
SELECT balance FROM accounts WHERE id = 1;  → 100

-- Transaction 2
UPDATE accounts SET balance = 200 WHERE id = 1;
COMMIT;

-- Transaction 1
SELECT balance FROM accounts WHERE id = 1;  → 200 (changed!)
```

**Phantom Read Example**:
```sql
-- Transaction 1
BEGIN;
SELECT COUNT(*) FROM users WHERE age > 30;  → 10

-- Transaction 2
INSERT INTO users (age) VALUES (35);
COMMIT;

-- Transaction 1
SELECT COUNT(*) FROM users WHERE age > 30;  → 11 (new row appeared!)
```

---

## 7. Query Optimization

### 7.1 EXPLAIN ANALYZE

```sql
EXPLAIN ANALYZE
SELECT * FROM users WHERE email = 'alice@example.com';
```

**Output**:
```
Seq Scan on users  (cost=0.00..1000.00 rows=1 width=100) (actual time=5.123..10.456 rows=1 loops=1)
  Filter: (email = 'alice@example.com')
  Rows Removed by Filter: 99999
Planning Time: 0.123 ms
Execution Time: 10.567 ms
```

**Key Metrics**:
- **Seq Scan**: Full table scan (slow)
- **cost**: Estimated cost (relative units)
- **actual time**: Real execution time
- **rows**: Rows processed

### 7.2 Understanding Query Plans

**Index Scan** (fast):
```
Index Scan using idx_users_email on users (cost=0.29..8.31 rows=1)
  Index Cond: (email = 'alice@example.com')
```

**Bitmap Scan** (multiple indexes):
```
Bitmap Heap Scan on users
  Recheck Cond: ((city = 'NYC') AND (age > 30))
  -> Bitmap Index Scan on idx_city
  -> Bitmap Index Scan on idx_age
```

**Nested Loop** (join):
```
Nested Loop (cost=0.00..100.00 rows=10)
  -> Seq Scan on users
  -> Index Scan on orders using idx_user_id
```

### 7.3 Optimization Techniques

#### Use Indexes
```sql
-- Bad: Full table scan
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';

-- Good: Use functional index
CREATE INDEX idx_email_lower ON users (LOWER(email));
SELECT * FROM users WHERE LOWER(email) = 'alice@example.com';
```

#### Avoid SELECT *
```sql
-- Bad
SELECT * FROM users WHERE id = 1;  -- Returns 50 columns

-- Good
SELECT id, name, email FROM users WHERE id = 1;  -- Only needed columns
```

#### Use EXISTS instead of IN for large subqueries
```sql
-- Slow for large subqueries
SELECT * FROM users WHERE id IN (SELECT user_id FROM orders);

-- Faster
SELECT * FROM users u WHERE EXISTS (SELECT 1 FROM orders o WHERE o.user_id = u.id);
```

---

## 8. Performance Tuning

### 8.1 Indexing Strategies

**B-Tree Index** (default):
```sql
CREATE INDEX idx_users_email ON users (email);
-- Good for: =, <, >, BETWEEN, ORDER BY
```

**Partial Index**:
```sql
CREATE INDEX idx_active_users ON users (email) WHERE status = 'active';
-- Only indexes active users (smaller, faster)
```

**Composite Index** (column order matters):
```sql
CREATE INDEX idx_users_city_age ON users (city, age);

-- Fast:
SELECT * FROM users WHERE city = 'NYC' AND age > 30;
SELECT * FROM users WHERE city = 'NYC';  -- Uses index

-- Slow (doesn't use index):
SELECT * FROM users WHERE age > 30;  -- city not in WHERE
```

**Covering Index** (includes extra columns):
```sql
CREATE INDEX idx_users_email_name ON users (email) INCLUDE (name);

-- Index-only scan (no table access)
SELECT name FROM users WHERE email = 'alice@example.com';
```

### 8.2 Partitioning

**Range Partitioning**:
```sql
CREATE TABLE orders (
    id BIGINT,
    created_at TIMESTAMP,
    amount NUMERIC
) PARTITION BY RANGE (created_at);

CREATE TABLE orders_2024_q1 PARTITION OF orders
    FOR VALUES FROM ('2024-01-01') TO ('2024-04-01');

CREATE TABLE orders_2024_q2 PARTITION OF orders
    FOR VALUES FROM ('2024-04-01') TO ('2024-07-01');
```

**Benefit**: Query only scans relevant partitions.
```sql
SELECT * FROM orders WHERE created_at BETWEEN '2024-01-01' AND '2024-01-31';
-- Only scans orders_2024_q1 partition
```

### 8.3 Connection Pooling

**Without pooling**:
```
Request 1: Open connection (100ms) → Query (10ms) → Close
Request 2: Open connection (100ms) → Query (10ms) → Close
Total: 220ms
```

**With pooling (PgBouncer)**:
```
Request 1: Reuse connection (0ms) → Query (10ms) → Return to pool
Request 2: Reuse connection (0ms) → Query (10ms) → Return to pool
Total: 20ms
```

### 8.4 Query Caching (Application-level)

```python
import redis
cache = redis.Redis()

def get_user(user_id):
    # Check cache first
    cached = cache.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss → query DB
    user = db.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    
    # Store in cache for 5 minutes
    cache.setex(f"user:{user_id}", 300, json.dumps(user))
    return user
```

---

## 9. Best Practices

### 9.1 Schema Design

✅ **Use appropriate data types**:
```sql
-- Bad: Wasting space
CREATE TABLE users (
    age VARCHAR(100)  -- Only need 0-120
);

-- Good
CREATE TABLE users (
    age SMALLINT CHECK (age >= 0 AND age <= 120)
);
```

✅ **Add constraints**:
```sql
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT NOT NULL REFERENCES users(id),
    amount NUMERIC CHECK (amount > 0),
    status VARCHAR(20) CHECK (status IN ('pending', 'completed', 'canceled')),
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 9.2 Query Best Practices

❌ **Don't use functions on indexed columns**:
```sql
-- Bad: Index not used
SELECT * FROM users WHERE YEAR(created_at) = 2024;

-- Good
SELECT * FROM users WHERE created_at >= '2024-01-01' AND created_at < '2025-01-01';
```

✅ **Use LIMIT**:
```sql
-- Bad: Returns all rows
SELECT * FROM logs ORDER BY created_at DESC;

-- Good
SELECT * FROM logs ORDER BY created_at DESC LIMIT 100;
```

### 9.3 Security

✅ **Use parameterized queries**:
```python
# Bad: SQL injection vulnerability
query = f"SELECT * FROM users WHERE email = '{email}'"

# Good
query = "SELECT * FROM users WHERE email = ?"
cursor.execute(query, (email,))
```

✅ **Principle of least privilege**:
```sql
-- Read-only user
CREATE USER readonly_user PASSWORD 'secure_password';
GRANT SELECT ON ALL TABLES IN SCHEMA public TO readonly_user;
```

---

## 10. Summary

### 10.1 Key Takeaways

1. ✅ **Joins** - Master INNER, LEFT, FULL OUTER
2. ✅ **CTEs** - Readable alternative to subqueries
3. ✅ **Window Functions** - Powerful for analytics
4. ✅ **Transactions** - ACID guarantees data integrity
5. ✅ **Indexing** - Critical for performance
6. ✅ **EXPLAIN** - Always analyze slow queries
7. ✅ **Parameterized Queries** - Prevent SQL injection

### 10.2 Performance Checklist

- [ ] Queries use indexes (check with EXPLAIN)
- [ ] SELECT only needed columns (not *)
- [ ] Connection pooling configured
- [ ] Transactions kept short
- [ ] Application-level caching for hot data
- [ ] Slow query log enabled

### 10.3 Tomorrow (Day 7): Normalization & Indexing

We'll deep dive into:
- **Normalization**: 1NF, 2NF, 3NF, BCNF
- **When to denormalize**
- **Index types**: B-Tree, Hash, GiST, GIN
- **Index maintenance**: VACUUM, ANALYZE
- **Monitoring**: pg_stat_statements

See you tomorrow! 🚀

---

**File Statistics**: ~1050 lines | Advanced SQL mastered ✅

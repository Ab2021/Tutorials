# Lab: Day 26 - SQL Performance Tuning

## Goal
Master `EXPLAIN ANALYZE`. You will generate a large dataset, run slow queries, and optimize them using Indexes.

## Prerequisites
- Docker (Postgres).

## Step 1: Setup Data
Connect to Postgres and run:

```sql
-- Create table
CREATE TABLE sales (
    id SERIAL PRIMARY KEY,
    product_id INT,
    customer_id INT,
    sale_date DATE,
    amount DECIMAL(10, 2)
);

-- Generate 1 Million Rows (Takes ~5-10s)
INSERT INTO sales (product_id, customer_id, sale_date, amount)
SELECT 
    floor(random() * 1000 + 1)::int,
    floor(random() * 50000 + 1)::int,
    '2023-01-01'::date + (random() * 365)::int,
    (random() * 100)::decimal(10, 2)
FROM generate_series(1, 1000000);
```

## Step 2: The Slow Query
Find total sales for a specific customer.

```sql
EXPLAIN ANALYZE SELECT SUM(amount) FROM sales WHERE customer_id = 12345;
```
*   **Observe**:
    *   `Seq Scan on sales`: It reads 1M rows.
    *   `Execution Time`: ~50-100ms (depending on hardware).

## Step 3: The Fix (B-Tree)

```sql
CREATE INDEX idx_customer ON sales(customer_id);
```

Run the query again:
```sql
EXPLAIN ANALYZE SELECT SUM(amount) FROM sales WHERE customer_id = 12345;
```
*   **Observe**:
    *   `Bitmap Heap Scan` or `Index Scan`.
    *   `Execution Time`: < 1ms.

## Step 4: Composite Index & Ordering

Query: Find sales for a customer on a specific date.
```sql
EXPLAIN ANALYZE SELECT * FROM sales WHERE customer_id = 12345 AND sale_date = '2023-06-01';
```

Create a composite index:
```sql
CREATE INDEX idx_cust_date ON sales(customer_id, sale_date);
```

**Test 1 (Match)**:
`WHERE customer_id = 12345 AND sale_date = '...'` -> Uses Index.

**Test 2 (Prefix)**:
`WHERE customer_id = 12345` -> Uses Index.

**Test 3 (Skip Prefix)**:
`WHERE sale_date = '...'` -> **Seq Scan** (Index not used!).

## Challenge: Covering Index
Optimize this query to be "Index Only Scan" (Zero Heap Access):
```sql
SELECT sale_date FROM sales WHERE customer_id = 12345;
```
*   *Hint*: `CREATE INDEX ... INCLUDE ...` or just add the column to the index.

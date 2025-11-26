# Lab: Day 7 - Indexes & Performance Tuning

## Goal
Witness the power of Indexes firsthand. You will generate a large dataset (100,000+ rows), run slow queries, analyze them with `EXPLAIN`, and then fix them with Indexes.

## Prerequisites
- Docker (Postgres container from Day 6).

## Step 1: Generate Data
We need enough data to make the DB sweat. We'll use SQL to generate dummy data.

Connect to your Postgres DB (`shop`) and run:

```sql
-- 1. Create a large table
CREATE TABLE big_users (
    id SERIAL PRIMARY KEY,
    email VARCHAR(100),
    age INT,
    country VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Insert 500,000 rows (This might take 10-20 seconds)
INSERT INTO big_users (email, age, country)
SELECT 
    'user_' || generate_series || '@example.com', 
    floor(random() * 80 + 18)::int, -- Age 18-98
    (array['USA', 'UK', 'India', 'Canada', 'Germany'])[floor(random() * 5 + 1)]
FROM generate_series(1, 500000);
```

## Step 2: The Slow Query (Seq Scan)

Run this query and note the time (in ms).
```sql
EXPLAIN ANALYZE SELECT * FROM big_users WHERE email = 'user_450000@example.com';
```
*   **Look for**: `Seq Scan on big_users`.
*   **Cost**: High. It checked 500k rows to find one.

## Step 3: The Fix (B-Tree Index)

Create an index on email.
```sql
CREATE INDEX idx_big_users_email ON big_users(email);
```

Run the query again:
```sql
EXPLAIN ANALYZE SELECT * FROM big_users WHERE email = 'user_450000@example.com';
```
*   **Look for**: `Index Scan using idx_big_users_email`.
*   **Result**: Should be instant (< 1ms).

## Step 4: Composite Indexes

Try this query:
```sql
EXPLAIN ANALYZE SELECT * FROM big_users WHERE country = 'India' AND age = 25;
```
It likely does a Seq Scan (or a Bitmap Scan if cardinality is low).

Create a composite index:
```sql
CREATE INDEX idx_country_age ON big_users(country, age);
```

Run it again. It should be much faster.

## Step 5: The "Index Miss"

Run this:
```sql
EXPLAIN ANALYZE SELECT * FROM big_users WHERE age = 25;
```
*   **Question**: Does it use `idx_country_age`?
*   **Answer**: No (or unlikely). Because `country` is the first column in the index. If you don't filter by country, the index is less useful (though Postgres has "Skip Scan" features in newer versions, usually it won't use it).

## Challenge: The Wildcard
Try to optimize this:
```sql
SELECT * FROM big_users WHERE email LIKE '%999%';
```
Standard indexes won't work. Look up `pg_trgm` extension if you want to solve this!

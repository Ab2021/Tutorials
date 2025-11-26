# Lab: Day 6 - SQL Bootcamp

## Goal
Master the basics of SQL. You will spin up a Postgres container, load a sample schema (E-commerce), and write queries to answer business questions.

## Prerequisites
- Docker.
- A SQL Client (DBeaver, TablePlus, or VS Code SQLTools extension).

## Directory Structure
```
day06/
├── docker-compose.yml
├── schema.sql
└── queries.sql (Your answers)
```

## Step 1: Docker Compose

```yaml
version: '3.8'
services:
  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: admin
      POSTGRES_PASSWORD: password
      POSTGRES_DB: shop
    ports:
      - "5432:5432"
    volumes:
      - ./schema.sql:/docker-entrypoint-initdb.d/init.sql
```

## Step 2: The Schema & Seed Data (`schema.sql`)

This file will run automatically when the container starts.

```sql
-- Users Table
CREATE TABLE users (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    email VARCHAR(100) UNIQUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Products Table
CREATE TABLE products (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100),
    price DECIMAL(10, 2),
    stock INT
);

-- Orders Table
CREATE TABLE orders (
    id SERIAL PRIMARY KEY,
    user_id INT REFERENCES users(id),
    total DECIMAL(10, 2),
    status VARCHAR(20) DEFAULT 'pending',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Order Items (Many-to-Many link)
CREATE TABLE order_items (
    order_id INT REFERENCES orders(id),
    product_id INT REFERENCES products(id),
    quantity INT,
    PRIMARY KEY (order_id, product_id)
);

-- Seed Data
INSERT INTO users (name, email) VALUES 
('Alice', 'alice@example.com'),
('Bob', 'bob@example.com'),
('Charlie', 'charlie@example.com');

INSERT INTO products (name, price, stock) VALUES 
('Laptop', 1000.00, 10),
('Mouse', 20.00, 100),
('Keyboard', 50.00, 50);

INSERT INTO orders (user_id, total, status) VALUES 
(1, 1020.00, 'shipped'), -- Alice bought Laptop + Mouse
(2, 50.00, 'pending');   -- Bob bought Keyboard

INSERT INTO order_items (order_id, product_id, quantity) VALUES 
(1, 1, 1), -- Laptop
(1, 2, 1), -- Mouse
(2, 3, 1); -- Keyboard
```

## Step 3: Run It
```bash
docker-compose up -d
```

## Step 4: The Challenge (Write these queries)

Connect to the DB (`localhost:5432`, user=`admin`, pass=`password`, db=`shop`) and write SQL for:

1.  **Basic Select**: List all products with price > $40.
2.  **Inner Join**: List all orders with the name of the user who placed them.
3.  **Aggregation**: Calculate the total revenue (sum of `total` in orders).
4.  **Complex Join**: List all items in Order #1 (Show Product Name, Quantity, and Price).
5.  **Group By**: Count how many orders each user has placed.
6.  **Left Join**: List all users and their orders (include users who have 0 orders).

## Solutions (Try not to peek!)
<details>
<summary>Click to see answers</summary>

```sql
-- 1. Products > $40
SELECT * FROM products WHERE price > 40;

-- 2. Orders with User Name
SELECT o.id, u.name, o.total 
FROM orders o 
JOIN users u ON o.user_id = u.id;

-- 3. Total Revenue
SELECT SUM(total) FROM orders;

-- 4. Items in Order #1
SELECT p.name, oi.quantity, p.price
FROM order_items oi
JOIN products p ON oi.product_id = p.id
WHERE oi.order_id = 1;

-- 5. Orders per User
SELECT u.name, COUNT(o.id) 
FROM users u 
LEFT JOIN orders o ON u.id = o.user_id 
GROUP BY u.id;

-- 6. All Users (Left Join)
SELECT u.name, o.id as order_id 
FROM users u 
LEFT JOIN orders o ON u.id = o.user_id;
```
</details>

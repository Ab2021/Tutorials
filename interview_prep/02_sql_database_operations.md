# SQL & Database Operations – Interview Prep
## Senior Data Scientist | General Mills & LSEG (India)

> **Target Roles:** Senior Data Scientist / Lead Data Scientist at General Mills (Mumbai/Bengaluru) and LSEG (Bengaluru/Hyderabad)
>
> **Interview Format:** Online Assessment (HackerRank/Codility) → Live coding on shared pads → Discussion of optimization choices
>
> **Key Databases Encountered:** PostgreSQL, Snowflake, BigQuery (General Mills), Oracle / SQL Server (LSEG)

---

## 📋 Table of Contents

1. [Fundamentals & Conceptual Questions](#1-fundamentals--conceptual-questions)
2. [Window Functions](#2-window-functions)
3. [CTEs & Subqueries](#3-ctes--subqueries)
4. [Complex Joins & Self-Joins](#4-complex-joins--self-joins)
5. [Aggregations & Conditional Logic](#5-aggregations--conditional-logic)
6. [Date & Time Operations](#6-date--time-operations)
7. [String Operations](#7-string-operations)
8. [Domain: FMCG / General Mills (Sales & Inventory)](#8-domain-fmcg--general-mills-sales--inventory)
9. [Domain: Financial / LSEG (Time Series & Trades)](#9-domain-financial--lseg-time-series--trades)
10. [Customer Segmentation (RFM)](#10-customer-segmentation-rfm)
11. [Gaps & Islands / Sessionization](#11-gaps--islands--sessionization)
12. [Performance Optimization](#12-performance-optimization)
13. [Python + SQL Integration](#13-python--sql-integration)
14. [Common Interview Gotchas](#14-common-interview-gotchas)

---

## Schema Reference

> These schemas are used across all questions below. Familiarise yourself with them.

```sql
-- FMCG / General Mills Schemas
CREATE TABLE products (
    product_id      INT PRIMARY KEY,
    product_name    VARCHAR(100),
    category        VARCHAR(50),
    brand           VARCHAR(50),
    unit_cost       DECIMAL(10,2),
    launch_date     DATE
);

CREATE TABLE sales (
    sale_id         BIGINT PRIMARY KEY,
    product_id      INT,
    retailer_id     INT,
    region          VARCHAR(50),
    sale_date       DATE,
    quantity        INT,
    unit_price      DECIMAL(10,2),
    discount_amount DECIMAL(10,2),
    is_promo        BOOLEAN
);

CREATE TABLE retailers (
    retailer_id     INT PRIMARY KEY,
    retailer_name   VARCHAR(100),
    channel         VARCHAR(50),   -- 'Modern Trade', 'General Trade', 'E-commerce'
    city            VARCHAR(50),
    state           VARCHAR(50)
);

CREATE TABLE inventory (
    inventory_id    BIGINT PRIMARY KEY,
    product_id      INT,
    warehouse_id    INT,
    stock_quantity  INT,
    recorded_date   DATE
);

-- Financial / LSEG Schemas
CREATE TABLE stock_prices (
    ticker          VARCHAR(20),
    trade_date      DATE,
    open_price      DECIMAL(12,4),
    close_price     DECIMAL(12,4),
    high_price      DECIMAL(12,4),
    low_price       DECIMAL(12,4),
    volume          BIGINT
);

CREATE TABLE trades (
    trade_id        BIGINT PRIMARY KEY,
    account_id      INT,
    ticker          VARCHAR(20),
    sector_id       INT,
    trade_type      VARCHAR(10),   -- 'BUY' or 'SELL'
    trade_time      TIMESTAMP,
    quantity        INT,
    price           DECIMAL(12,4),
    amount          DECIMAL(14,2)
);
```

---

## 1. Fundamentals & Conceptual Questions

### Q1 — WHERE vs HAVING 🟢 Easy

> **Asked at:** General Mills OA (Online Assessment). Classic warm-up filter question.

**Question:** Explain the difference between `WHERE` and `HAVING`. When would you use each?

**Answer:**

| Clause | Filters On | Operates On |
|--------|-----------|-------------|
| `WHERE` | Individual row conditions | Raw table data *before* `GROUP BY` |
| `HAVING` | Aggregate results | Grouped data *after* `GROUP BY` |

```sql
-- WHERE: Filter raw rows before aggregation
SELECT region, SUM(quantity * unit_price) AS total_revenue
FROM sales
WHERE sale_date >= '2024-01-01'          -- Only 2024 sales go into the aggregate
GROUP BY region;

-- HAVING: Filter after aggregation
SELECT region, SUM(quantity * unit_price) AS total_revenue
FROM sales
GROUP BY region
HAVING SUM(quantity * unit_price) > 1000000;   -- Only high-revenue regions

-- Combined: Both together
SELECT region, SUM(quantity * unit_price) AS total_revenue
FROM sales
WHERE sale_date >= '2024-01-01'          -- Pre-filter rows
GROUP BY region
HAVING SUM(quantity * unit_price) > 500000;    -- Post-filter groups
```

**Gotcha:** You cannot use a column alias defined in `SELECT` inside `HAVING` in most SQL dialects (use the full expression instead).

---

### Q2 — JOIN Types & NULL Behaviour 🟢 Easy

> **Asked at:** General Mills Round 2 (Technical Screen). Every FMCG analytics team joins products ↔ sales extensively.

**Question:** What is the difference between `INNER JOIN`, `LEFT JOIN`, and `FULL OUTER JOIN`? Which would you use to find products that have never been sold?

```sql
-- Products that have NEVER appeared in sales (LEFT JOIN anti-pattern)
SELECT p.product_id, p.product_name, p.category
FROM products p
LEFT JOIN sales s ON p.product_id = s.product_id
WHERE s.sale_id IS NULL;        -- NULL means no matching sale row
-- Note: Use IS NULL, not = NULL

-- FULL OUTER JOIN: See ALL products and ALL sales, even unmatched on both sides
SELECT p.product_name, s.sale_id, s.sale_date
FROM products p
FULL OUTER JOIN sales s ON p.product_id = s.product_id;

-- INNER JOIN: Only matched rows (most common, most restrictive)
SELECT p.product_name, SUM(s.quantity) AS units_sold
FROM products p
INNER JOIN sales s ON p.product_id = s.product_id
GROUP BY p.product_name;
```

**Interview Insight:** Always clarify whether "all products" is required (LEFT JOIN) or only "products with sales" (INNER JOIN). Misusing INNER JOIN silently excludes new products with zero sales.

---

### Q3 — RANK vs DENSE_RANK vs ROW_NUMBER 🟡 Medium

> **Asked at:** General Mills & LSEG — almost universally the first window function question.

**Question:** What is the difference between `RANK()`, `DENSE_RANK()`, and `ROW_NUMBER()`? Demonstrate with an example.

```sql
-- Given: Product sales data with ties
WITH product_sales AS (
    SELECT
        p.product_name,
        p.category,
        SUM(s.quantity * s.unit_price) AS total_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_name, p.category
)
SELECT
    product_name,
    category,
    total_revenue,
    ROW_NUMBER()   OVER (PARTITION BY category ORDER BY total_revenue DESC) AS row_num,
    -- ROW_NUMBER: Always unique; 1,2,3,4 even on ties (arbitrary tiebreak)

    RANK()         OVER (PARTITION BY category ORDER BY total_revenue DESC) AS rnk,
    -- RANK: Ties get the same rank; GAPS after: 1,2,2,4 (skips 3)

    DENSE_RANK()   OVER (PARTITION BY category ORDER BY total_revenue DESC) AS dense_rnk
    -- DENSE_RANK: Ties get same rank; NO gaps: 1,2,2,3
FROM product_sales;

/*
Example output for category = 'Snacks':
product_name  | total_revenue | row_num | rnk | dense_rnk
Lays          | 500000        |    1    |  1  |     1
Kurkure       | 450000        |    2    |  2  |     2
Pringles      | 450000        |    3    |  2  |     2   <-- tie
Doritos       | 300000        |    4    |  4  |     3   <-- RANK skips 3, DENSE_RANK does not
*/
```

**When to use which:**
- `ROW_NUMBER` → When you need a strict unique identifier (pagination, deduplication)
- `RANK` → When ties should leave gaps (competition-style ranking)
- `DENSE_RANK` → When ties should NOT leave gaps (most common in DS interviews)

---

## 2. Window Functions

### Q4 — 7-Day Moving Average of Sales 🟡 Medium

> **Asked at:** General Mills (demand forecasting context) and LSEG (stock price smoothing). Extremely common.

**Question:** Calculate a 7-day moving average of daily revenue for each product category.

```sql
-- Step 1: Compute daily revenue per category
WITH daily_revenue AS (
    SELECT
        p.category,
        s.sale_date,
        SUM(s.quantity * s.unit_price) AS daily_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category, s.sale_date
)
-- Step 2: Apply moving average window
SELECT
    category,
    sale_date,
    daily_revenue,
    ROUND(
        AVG(daily_revenue) OVER (
            PARTITION BY category
            ORDER BY sale_date
            ROWS BETWEEN 6 PRECEDING AND CURRENT ROW  -- 7 rows incl. today
        ), 2
    ) AS moving_avg_7d,
    -- Also compute 30-day moving average for broader trend
    ROUND(
        AVG(daily_revenue) OVER (
            PARTITION BY category
            ORDER BY sale_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ), 2
    ) AS moving_avg_30d
FROM daily_revenue
ORDER BY category, sale_date;

/*
Interviewer follow-up: "What if there are missing dates (no sales on some days)?"
Answer: The ROWS BETWEEN uses physical rows, so gaps are silently skipped.
To be precise, you'd need a date spine (calendar table) joined in.
*/
```

---

### Q5 — LAG and LEAD: Month-over-Month Growth 🟡 Medium

> **Asked at:** General Mills and LSEG. Essential for time-series analysis.

**Question:** Calculate month-over-month revenue growth (%) per region using `LAG`.

```sql
WITH monthly_revenue AS (
    SELECT
        region,
        DATE_TRUNC('month', sale_date)       AS sales_month,
        SUM(quantity * unit_price)            AS current_revenue
    FROM sales
    GROUP BY region, DATE_TRUNC('month', sale_date)
)
SELECT
    region,
    sales_month,
    current_revenue,
    LAG(current_revenue) OVER (
        PARTITION BY region
        ORDER BY sales_month
    )                                         AS prev_month_revenue,

    -- MoM Growth %: (current - previous) / previous * 100
    ROUND(
        (current_revenue
            - LAG(current_revenue) OVER (PARTITION BY region ORDER BY sales_month))
        / NULLIF(LAG(current_revenue) OVER (PARTITION BY region ORDER BY sales_month), 0)
        * 100,
    2)                                        AS mom_growth_pct,

    -- LEAD: What's NEXT month's revenue (useful for forecasting comparison)
    LEAD(current_revenue) OVER (
        PARTITION BY region
        ORDER BY sales_month
    )                                         AS next_month_revenue
FROM monthly_revenue
ORDER BY region, sales_month;

/*
Key: Use NULLIF(prev, 0) to avoid division-by-zero errors.
First month row will have NULL for prev_month_revenue and mom_growth_pct — expected behaviour.
*/
```

---

### Q6 — Top N per Group with DENSE_RANK 🟡 Medium

> **Asked at:** General Mills (top products per region), LSEG (top trades per sector). Asked in almost every Senior DS interview.

**Question:** Find the top 3 products by total revenue in each category.

```sql
WITH product_revenue AS (
    -- Aggregate total revenue per product and category
    SELECT
        p.category,
        p.product_name,
        SUM(s.quantity * s.unit_price)        AS total_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.category, p.product_name
),
ranked AS (
    -- Rank within each category by revenue descending
    SELECT
        category,
        product_name,
        total_revenue,
        DENSE_RANK() OVER (
            PARTITION BY category
            ORDER BY total_revenue DESC
        )                                     AS rnk
    FROM product_revenue
)
-- Filter to top 3 only
SELECT category, product_name, total_revenue, rnk
FROM ranked
WHERE rnk <= 3
ORDER BY category, rnk;

/*
Why DENSE_RANK vs RANK?
  RANK: If two products tie at rank 2, the next one is ranked 4 — you'd miss it with rnk <= 3
  DENSE_RANK: Ties get same rank, next is 3 — you always get "top 3 unique revenue levels"
  INTERVIEWER TIP: Always clarify whether tied products should BOTH appear in top-3
*/
```

---

### Q7 — Running Total and Cumulative Share (Pareto / 80-20) 🔴 Hard

> **Asked at:** General Mills Finance & Analytics teams for identifying hero SKUs.

**Question:** Find which products contribute to the top 80% of total revenue (Pareto principle). Flag them as "Core SKU" vs "Tail SKU".

```sql
WITH product_revenue AS (
    SELECT
        p.product_id,
        p.product_name,
        p.category,
        SUM(s.quantity * s.unit_price)                     AS total_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name, p.category
),
running_totals AS (
    SELECT
        product_id,
        product_name,
        category,
        total_revenue,
        -- Cumulative revenue from highest to lowest
        SUM(total_revenue) OVER (
            ORDER BY total_revenue DESC
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                   AS cum_revenue,
        -- Grand total for percentage calculation
        SUM(total_revenue) OVER ()                          AS grand_total
    FROM product_revenue
)
SELECT
    product_id,
    product_name,
    category,
    total_revenue,
    ROUND(total_revenue / grand_total * 100, 2)             AS revenue_share_pct,
    ROUND(cum_revenue   / grand_total * 100, 2)             AS cum_share_pct,
    -- Flag: a product is "Core SKU" if it falls within the cumulative 80% threshold
    -- Use cum_revenue MINUS its own contribution to check boundary inclusively
    CASE
        WHEN (cum_revenue - total_revenue) / grand_total < 0.80
        THEN 'Core SKU (Top 80%)'
        ELSE 'Tail SKU'
    END                                                     AS sku_segment
FROM running_totals
ORDER BY total_revenue DESC;
```

---

### Q8 — NTILE for Quartile Bucketing 🟡 Medium

> **Asked at:** LSEG (volume distribution buckets), General Mills (performance deciles).

**Question:** Bucket products into revenue quartiles (Q1 = bottom 25%, Q4 = top 25%).

```sql
WITH product_revenue AS (
    SELECT
        p.product_id,
        p.product_name,
        SUM(s.quantity * s.unit_price) AS total_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY p.product_id, p.product_name
)
SELECT
    product_id,
    product_name,
    total_revenue,
    NTILE(4) OVER (ORDER BY total_revenue ASC)   AS revenue_quartile,
    -- 1 = lowest 25%, 4 = highest 25%
    NTILE(10) OVER (ORDER BY total_revenue ASC)  AS revenue_decile
FROM product_revenue
ORDER BY total_revenue DESC;
```

---

## 3. CTEs & Subqueries

### Q9 — Recursive CTE: Hierarchical Data 🔴 Hard

> **Asked at:** LSEG (org chart / entity hierarchy), General Mills (category hierarchy).

**Question:** Given a product category hierarchy (parent-child), find all products under the "Snacks" top-level category, at any depth.

```sql
-- Table: category_hierarchy(category_id, category_name, parent_category_id)
WITH RECURSIVE category_tree AS (
    -- Base case: start from the root 'Snacks' category
    SELECT
        category_id,
        category_name,
        parent_category_id,
        0                   AS depth,
        category_name       AS path
    FROM category_hierarchy
    WHERE category_name = 'Snacks'

    UNION ALL

    -- Recursive case: join children to current level
    SELECT
        ch.category_id,
        ch.category_name,
        ch.parent_category_id,
        ct.depth + 1,
        ct.path || ' > ' || ch.category_name
    FROM category_hierarchy ch
    INNER JOIN category_tree ct ON ch.parent_category_id = ct.category_id
)
SELECT
    category_id,
    category_name,
    depth,
    path
FROM category_tree
ORDER BY path;

/*
Result includes: Snacks > Chips > Potato Chips, Snacks > Biscuits, etc.
Useful for roll-up reporting across a product hierarchy.
*/
```

---

### Q10 — Multi-Step CTE: Cohort Retention Analysis 🔴 Hard

> **Asked at:** General Mills (consumer analytics), common at all Senior DS roles.

**Question:** Calculate monthly retention rates: for each cohort (month of first purchase), what percentage returned to buy in subsequent months?

```sql
WITH first_purchase AS (
    -- Step 1: Identify each customer's first purchase month (cohort)
    SELECT
        retailer_id,
        MIN(DATE_TRUNC('month', sale_date))   AS cohort_month
    FROM sales
    GROUP BY retailer_id
),
monthly_activity AS (
    -- Step 2: Get every active month per customer
    SELECT DISTINCT
        s.retailer_id,
        DATE_TRUNC('month', s.sale_date)      AS activity_month
    FROM sales s
),
cohort_data AS (
    -- Step 3: Join cohort info and compute month number (0 = cohort month, 1 = next month, ...)
    SELECT
        fp.cohort_month,
        ma.activity_month,
        EXTRACT(YEAR FROM AGE(ma.activity_month, fp.cohort_month)) * 12
            + EXTRACT(MONTH FROM AGE(ma.activity_month, fp.cohort_month)) AS months_since_start,
        ma.retailer_id
    FROM first_purchase fp
    JOIN monthly_activity ma ON fp.retailer_id = ma.retailer_id
),
cohort_size AS (
    -- Step 4: Count cohort size (month 0 users)
    SELECT cohort_month, COUNT(DISTINCT retailer_id) AS cohort_count
    FROM cohort_data
    WHERE months_since_start = 0
    GROUP BY cohort_month
)
-- Step 5: Compute retention rate per cohort and month number
SELECT
    cd.cohort_month,
    cd.months_since_start,
    COUNT(DISTINCT cd.retailer_id)                    AS active_customers,
    cs.cohort_count,
    ROUND(
        COUNT(DISTINCT cd.retailer_id)::DECIMAL
        / cs.cohort_count * 100, 1
    )                                                 AS retention_pct
FROM cohort_data cd
JOIN cohort_size cs ON cd.cohort_month = cs.cohort_month
GROUP BY cd.cohort_month, cd.months_since_start, cs.cohort_count
ORDER BY cd.cohort_month, cd.months_since_start;
```

---

## 4. Complex Joins & Self-Joins

### Q11 — Self-Join: Find Consecutive Events Within a Time Window 🔴 Hard

> **Asked at:** LSEG (rapid trading activity detection / compliance). Very domain-specific.

**Question:** Find all pairs of trades by the same account made within 5 minutes of each other (potential wash trading or high-frequency activity).

```sql
SELECT
    a.account_id,
    a.trade_id      AS trade_1_id,
    b.trade_id      AS trade_2_id,
    a.ticker,
    a.trade_time    AS time_1,
    b.trade_time    AS time_2,
    -- Time difference in minutes
    EXTRACT(EPOCH FROM (b.trade_time - a.trade_time)) / 60.0 AS gap_minutes,
    a.amount        AS amount_1,
    b.amount        AS amount_2
FROM trades a
JOIN trades b
    ON  a.account_id = b.account_id
    AND a.trade_id   < b.trade_id              -- Prevent duplicate pairs and self-join
    AND b.trade_time >= a.trade_time
    AND b.trade_time <= a.trade_time + INTERVAL '5 minutes'
ORDER BY a.account_id, a.trade_time;

/*
Key design choices:
1. a.trade_id < b.trade_id: ensures (A,B) but not (B,A) is returned — avoids duplicate pairs
2. The join condition on time window is a non-equijoin (range join)
3. Performance tip: Index on (account_id, trade_time) is critical at LSEG scale
*/
```

---

### Q12 — Multi-Table Join: Sales Dashboard 🟡 Medium

> **Asked at:** General Mills Round 3. Tests ability to join normalized schemas cleanly.

**Question:** Build a sales performance report showing revenue, units, discount %, and promotion flag per region, channel, and category for Q1 2024.

```sql
SELECT
    s.region,
    r.channel,
    p.category,
    COUNT(DISTINCT s.sale_id)                               AS total_transactions,
    SUM(s.quantity)                                         AS total_units,
    SUM(s.quantity * s.unit_price)                          AS gross_revenue,
    SUM(s.discount_amount)                                  AS total_discounts,
    SUM(s.quantity * s.unit_price - s.discount_amount)      AS net_revenue,
    ROUND(
        SUM(s.discount_amount)
        / NULLIF(SUM(s.quantity * s.unit_price), 0) * 100,
    1)                                                      AS discount_pct,
    SUM(CASE WHEN s.is_promo THEN s.quantity * s.unit_price ELSE 0 END)
                                                            AS promo_revenue,
    SUM(CASE WHEN NOT s.is_promo THEN s.quantity * s.unit_price ELSE 0 END)
                                                            AS base_revenue
FROM sales s
JOIN products  p ON s.product_id  = p.product_id
JOIN retailers r ON s.retailer_id = r.retailer_id
WHERE s.sale_date BETWEEN '2024-01-01' AND '2024-03-31'
GROUP BY s.region, r.channel, p.category
ORDER BY net_revenue DESC;
```

---

## 5. Aggregations & Conditional Logic

### Q13 — Pivot with CASE WHEN: Cross-Tab Revenue by Quarter 🟡 Medium

> **Asked at:** General Mills Finance. Pivot tables are essential in CPG analytics.

**Question:** Create a pivot showing revenue by quarter (columns: Q1, Q2, Q3, Q4) for each product category in 2024.

```sql
SELECT
    p.category,
    SUM(CASE WHEN EXTRACT(QUARTER FROM s.sale_date) = 1
             THEN s.quantity * s.unit_price ELSE 0 END)   AS q1_revenue,
    SUM(CASE WHEN EXTRACT(QUARTER FROM s.sale_date) = 2
             THEN s.quantity * s.unit_price ELSE 0 END)   AS q2_revenue,
    SUM(CASE WHEN EXTRACT(QUARTER FROM s.sale_date) = 3
             THEN s.quantity * s.unit_price ELSE 0 END)   AS q3_revenue,
    SUM(CASE WHEN EXTRACT(QUARTER FROM s.sale_date) = 4
             THEN s.quantity * s.unit_price ELSE 0 END)   AS q4_revenue,
    SUM(s.quantity * s.unit_price)                        AS annual_revenue
FROM sales s
JOIN products p ON s.product_id = p.product_id
WHERE EXTRACT(YEAR FROM s.sale_date) = 2024
GROUP BY p.category
ORDER BY annual_revenue DESC;
```

---

### Q14 — Finding and Deduplicating Duplicate Records 🟡 Medium

> **Asked at:** General Mills (data quality checks in ETL pipelines).

**Question:** Identify and remove duplicate sales records (same product, retailer, date, and quantity — keep the row with the smallest sale_id).

```sql
-- Step 1: Identify duplicates
WITH duplicate_check AS (
    SELECT
        sale_id,
        product_id,
        retailer_id,
        sale_date,
        quantity,
        ROW_NUMBER() OVER (
            PARTITION BY product_id, retailer_id, sale_date, quantity
            ORDER BY sale_id ASC        -- keep the first (lowest) sale_id
        ) AS rn
    FROM sales
)
-- Step 2: See which are duplicates
SELECT * FROM duplicate_check WHERE rn > 1;

-- Step 3: Delete duplicates (keep rn = 1)
DELETE FROM sales
WHERE sale_id IN (
    SELECT sale_id
    FROM duplicate_check
    WHERE rn > 1
);

/*
Alternative (non-destructive): Create a deduplicated view
CREATE VIEW sales_deduped AS
SELECT * FROM duplicate_check WHERE rn = 1;
*/
```

---

## 6. Date & Time Operations

### Q15 — Date Spine & Filling Missing Dates 🔴 Hard

> **Asked at:** General Mills (demand forecasting — gaps in daily sales must be filled with 0).

**Question:** Generate a date spine for a full year and fill in missing daily sales with 0.

```sql
-- PostgreSQL: Generate a continuous date range
WITH date_spine AS (
    SELECT
        generate_series(
            '2024-01-01'::DATE,
            '2024-12-31'::DATE,
            '1 day'::INTERVAL
        )::DATE AS sales_date
),
category_spine AS (
    -- Cross-join dates × categories to get all combinations
    SELECT ds.sales_date, p.category
    FROM date_spine ds
    CROSS JOIN (SELECT DISTINCT category FROM products) p
),
daily_sales AS (
    SELECT
        DATE_TRUNC('day', s.sale_date)::DATE AS sales_date,
        p.category,
        SUM(s.quantity * s.unit_price)       AS daily_revenue
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    GROUP BY DATE_TRUNC('day', s.sale_date)::DATE, p.category
)
-- Left join: fills NULL revenue for missing dates with COALESCE
SELECT
    cs.sales_date,
    cs.category,
    COALESCE(ds.daily_revenue, 0)            AS daily_revenue
FROM category_spine cs
LEFT JOIN daily_sales ds
    ON cs.sales_date = cs.sales_date
    AND cs.category  = ds.category
ORDER BY cs.category, cs.sales_date;
```

---

### Q16 — Age, Tenure, and Date Arithmetic 🟢 Easy

> **Asked at:** General Mills HR/People analytics, LSEG compliance reporting.

**Question:** Identify products launched more than 2 years ago that have had no sales in the last 6 months (potential discontinuation candidates).

```sql
SELECT
    p.product_id,
    p.product_name,
    p.category,
    p.launch_date,
    -- Product age in months
    EXTRACT(MONTH FROM AGE(CURRENT_DATE, p.launch_date))
    + EXTRACT(YEAR  FROM AGE(CURRENT_DATE, p.launch_date)) * 12  AS age_months,
    MAX(s.sale_date)                                              AS last_sale_date,
    CURRENT_DATE - MAX(s.sale_date)                              AS days_since_last_sale
FROM products p
LEFT JOIN sales s ON p.product_id = s.product_id
WHERE p.launch_date <= CURRENT_DATE - INTERVAL '2 years'
GROUP BY p.product_id, p.product_name, p.category, p.launch_date
HAVING MAX(s.sale_date) < CURRENT_DATE - INTERVAL '6 months'
    OR MAX(s.sale_date) IS NULL        -- products with zero sales ever
ORDER BY last_sale_date ASC NULLS FIRST;
```

---

## 7. String Operations

### Q17 — String Parsing & Cleaning 🟡 Medium

> **Asked at:** General Mills data ingestion pipelines. Messy data is a reality in CPG.

**Question:** From a raw `product_code` field like `"GM-SNK-001-IND"`, extract brand code, category code, SKU number, and market. Also standardise retailer names (trim whitespace, uppercase).

```sql
SELECT
    product_code,
    -- Split on '-': positions 1 to 4
    SPLIT_PART(product_code, '-', 1)             AS brand_code,     -- 'GM'
    SPLIT_PART(product_code, '-', 2)             AS category_code,  -- 'SNK'
    SPLIT_PART(product_code, '-', 3)             AS sku_number,     -- '001'
    SPLIT_PART(product_code, '-', 4)             AS market,         -- 'IND'

    -- String length
    LENGTH(product_code)                          AS code_length,

    -- Pattern match: products for India market
    CASE
        WHEN product_code LIKE '%-IND'           THEN 'India'
        WHEN product_code LIKE '%-US'            THEN 'US'
        ELSE 'Other'
    END                                           AS market_label,

    -- Clean retailer name: trim and standardise
    UPPER(TRIM(retailer_name))                    AS clean_retailer_name,

    -- Extract year from product_code embedded date if format: 'GM-SNK-001-2024'
    CASE
        WHEN SPLIT_PART(product_code, '-', 4) ~ '^\d{4}$'
        THEN CAST(SPLIT_PART(product_code, '-', 4) AS INT)
        ELSE NULL
    END                                           AS launch_year
FROM products
-- Filter for India SKUs with ILIKE (case-insensitive)
WHERE product_code ILIKE '%-IND';
```

---

## 8. Domain: FMCG / General Mills (Sales & Inventory)

### Q18 — Inventory: Days of Inventory on Hand (DOI) 🟡 Medium

> **Very domain-specific to General Mills supply chain analytics.**

**Question:** Calculate Days of Inventory (DOI) for each product category — how many days can current stock sustain current demand?

```sql
WITH daily_cogs AS (
    -- Average daily units sold over the last 30 days per category
    SELECT
        p.category,
        SUM(s.quantity * p.unit_cost)
        / NULLIF(COUNT(DISTINCT s.sale_date), 0)  AS avg_daily_cogs
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    WHERE s.sale_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY p.category
),
current_inventory AS (
    -- Latest inventory snapshot (most recent recorded_date)
    SELECT
        p.category,
        SUM(i.stock_quantity * p.unit_cost)       AS total_inventory_value
    FROM inventory i
    JOIN products p ON i.product_id = p.product_id
    WHERE i.recorded_date = (SELECT MAX(recorded_date) FROM inventory)
    GROUP BY p.category
)
SELECT
    ci.category,
    ROUND(ci.total_inventory_value, 0)            AS inventory_value_inr,
    ROUND(dc.avg_daily_cogs, 0)                   AS avg_daily_cogs,
    CASE
        WHEN dc.avg_daily_cogs > 0
        THEN ROUND(ci.total_inventory_value / dc.avg_daily_cogs, 1)
        ELSE NULL
    END                                           AS days_of_inventory,
    CASE
        WHEN ci.total_inventory_value / NULLIF(dc.avg_daily_cogs, 0) < 7
        THEN '🔴 Critical'
        WHEN ci.total_inventory_value / NULLIF(dc.avg_daily_cogs, 0) < 30
        THEN '🟡 Low'
        ELSE '🟢 Healthy'
    END                                           AS inventory_status
FROM current_inventory ci
LEFT JOIN daily_cogs dc ON ci.category = dc.category
ORDER BY days_of_inventory ASC NULLS FIRST;
```

---

### Q19 — Trade Promotion ROI Analysis 🔴 Hard

> **Asked at:** General Mills Commercial Finance. Promotions can erode margins if not analysed correctly.

**Question:** Compare net profit for promoted vs non-promoted transactions. Calculate incremental volume and uplift due to promotions.

```sql
WITH promo_summary AS (
    SELECT
        p.category,
        s.is_promo,
        COUNT(DISTINCT s.sale_id)                             AS total_transactions,
        SUM(s.quantity)                                       AS total_units,
        SUM(s.quantity * s.unit_price)                        AS gross_revenue,
        SUM(s.discount_amount)                                AS total_discounts,
        SUM(s.quantity * s.unit_price - s.discount_amount)    AS net_revenue,
        SUM(s.quantity * (s.unit_price - p.unit_cost)
            - s.discount_amount)                              AS net_profit,
        -- Average selling price
        ROUND(SUM(s.quantity * s.unit_price)
              / NULLIF(SUM(s.quantity), 0), 2)                AS avg_selling_price
    FROM sales s
    JOIN products p ON s.product_id = p.product_id
    WHERE s.sale_date BETWEEN '2024-01-01' AND '2024-12-31'
    GROUP BY p.category, s.is_promo
)
SELECT
    ps_promo.category,
    ps_promo.total_units                           AS promo_units,
    ps_base.total_units                            AS base_units,
    -- Incremental volume driven by promotion
    ps_promo.total_units - ps_base.total_units     AS incremental_units,
    ROUND(
        (ps_promo.total_units - ps_base.total_units)
        / NULLIF(ps_base.total_units, 0) * 100, 1
    )                                              AS volume_uplift_pct,
    ps_promo.net_profit                            AS promo_net_profit,
    ps_base.net_profit                             AS base_net_profit,
    ps_promo.net_profit - ps_base.net_profit       AS incremental_profit
FROM promo_summary ps_promo
JOIN promo_summary ps_base
    ON  ps_promo.category = ps_base.category
    AND ps_promo.is_promo  = TRUE
    AND ps_base.is_promo   = FALSE
ORDER BY incremental_profit DESC;
```

---

## 9. Domain: Financial / LSEG (Time Series & Trades)

### Q20 — Stock Price: 7-Day Moving Average & Bollinger Bands 🔴 Hard

> **Core LSEG interview question. Financial time-series with window functions.**

**Question:** Calculate a 20-day moving average and Bollinger Bands (MA ± 2 standard deviations) for each stock ticker.

```sql
WITH moving_stats AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        -- 20-day simple moving average
        AVG(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )                                             AS sma_20,
        -- 20-day standard deviation
        STDDEV(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )                                             AS stddev_20,
        COUNT(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 19 PRECEDING AND CURRENT ROW
        )                                             AS window_size     -- ensure full 20-day window
    FROM stock_prices
)
SELECT
    ticker,
    trade_date,
    close_price,
    ROUND(sma_20, 4)                                  AS sma_20,
    -- Upper Bollinger Band
    ROUND(sma_20 + 2 * stddev_20, 4)                  AS upper_band,
    -- Lower Bollinger Band
    ROUND(sma_20 - 2 * stddev_20, 4)                  AS lower_band,
    -- Signal: price outside Bollinger Bands
    CASE
        WHEN close_price > sma_20 + 2 * stddev_20     THEN 'OVERBOUGHT'
        WHEN close_price < sma_20 - 2 * stddev_20     THEN 'OVERSOLD'
        ELSE 'NEUTRAL'
    END                                               AS signal
FROM moving_stats
WHERE window_size = 20              -- Only show rows with a full 20-day window
ORDER BY ticker, trade_date;
```

---

### Q21 — Daily Returns & Volatility 🔴 Hard

> **LSEG — Essential financial metric. Tests LAG and mathematical computation in SQL.**

**Question:** Calculate daily returns, 30-day rolling volatility (annualised), and identify days with abnormal price movement (>2 sigma).

```sql
WITH daily_returns AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        LAG(close_price) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
        )                                             AS prev_close,
        -- Daily log return (more statistically stable than simple return)
        LN(close_price / NULLIF(
            LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date),
        0))                                           AS log_return
    FROM stock_prices
),
rolling_volatility AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        log_return,
        -- 30-day rolling mean and stddev of returns
        AVG(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                             AS rolling_mean_return,
        STDDEV(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        )                                             AS rolling_stddev,
        -- Annualised volatility: stddev × sqrt(252 trading days)
        STDDEV(log_return) OVER (
            PARTITION BY ticker
            ORDER BY trade_date
            ROWS BETWEEN 29 PRECEDING AND CURRENT ROW
        ) * SQRT(252)                                 AS annualised_volatility
    FROM daily_returns
    WHERE log_return IS NOT NULL
)
SELECT
    ticker,
    trade_date,
    ROUND(close_price, 2)                             AS close_price,
    ROUND(log_return * 100, 4)                        AS daily_return_pct,
    ROUND(annualised_volatility * 100, 2)             AS annual_vol_pct,
    -- Z-score: how many sigmas from rolling mean
    ROUND(
        (log_return - rolling_mean_return)
        / NULLIF(rolling_stddev, 0),
    2)                                                AS return_zscore,
    CASE
        WHEN ABS((log_return - rolling_mean_return)
                 / NULLIF(rolling_stddev, 0)) > 2
        THEN 'ABNORMAL'
        ELSE 'NORMAL'
    END                                               AS movement_flag
FROM rolling_volatility
ORDER BY ticker, trade_date;
```

---

## 10. Customer Segmentation (RFM)

### Q22 — Full RFM Scoring with NTILE 🔴 Hard

> **Asked at:** General Mills Consumer Analytics, LSEG Client Intelligence. Classic DS question.

**Question:** Build an RFM (Recency, Frequency, Monetary) model to segment retailers into tiers (Champions, At Risk, Lost, etc.).

```sql
WITH rfm_raw AS (
    SELECT
        retailer_id,
        -- Recency: days since last purchase (lower = more recent = better)
        CURRENT_DATE - MAX(sale_date)                   AS recency_days,
        -- Frequency: number of distinct purchase occasions
        COUNT(DISTINCT sale_date)                       AS frequency,
        -- Monetary: total spend
        SUM(quantity * unit_price - discount_amount)    AS monetary
    FROM sales
    GROUP BY retailer_id
),
rfm_scored AS (
    SELECT
        retailer_id,
        recency_days,
        frequency,
        monetary,
        -- Recency score: 5 = most recent (invert: lower recency_days → higher score)
        6 - NTILE(5) OVER (ORDER BY recency_days ASC)  AS r_score,
        NTILE(5) OVER (ORDER BY frequency ASC)         AS f_score,
        NTILE(5) OVER (ORDER BY monetary ASC)          AS m_score
    FROM rfm_raw
),
rfm_segments AS (
    SELECT
        retailer_id,
        recency_days,
        frequency,
        ROUND(monetary, 0)                              AS monetary,
        r_score,
        f_score,
        m_score,
        (r_score + f_score + m_score)                   AS rfm_total,
        -- Segment based on combined score pattern
        CASE
            WHEN r_score >= 4 AND f_score >= 4          THEN 'Champion'
            WHEN r_score >= 3 AND f_score >= 3          THEN 'Loyal Customer'
            WHEN r_score >= 4 AND f_score <= 2          THEN 'Recent Buyer'
            WHEN r_score <= 2 AND f_score >= 3          THEN 'At Risk'
            WHEN r_score <= 2 AND f_score <= 2
                 AND m_score >= 3                       THEN 'Cannot Lose Them'
            WHEN r_score <= 1                           THEN 'Lost'
            ELSE 'Needs Attention'
        END                                             AS rfm_segment
    FROM rfm_scored
)
SELECT
    rfm_segment,
    COUNT(*)                                            AS retailer_count,
    ROUND(AVG(recency_days), 0)                         AS avg_recency_days,
    ROUND(AVG(frequency), 1)                            AS avg_frequency,
    ROUND(AVG(monetary), 0)                             AS avg_monetary,
    ROUND(SUM(monetary), 0)                             AS total_monetary
FROM rfm_segments
GROUP BY rfm_segment
ORDER BY total_monetary DESC;
```

---

## 11. Gaps & Islands / Sessionization

### Q23 — Classic Gaps & Islands: Market Closure Streaks 🔴 Hard

> **Asked at:** LSEG (trading halt analysis, compliance). The canonical advanced SQL pattern.

**Question:** Find continuous streaks where the market was 'CLOSED' — report start date, end date, and length of each streak.

```sql
-- Table: market_status(status_date DATE, status VARCHAR)
WITH closed_days AS (
    -- Filter only closed days
    SELECT
        status_date,
        -- Global row number vs partition row number trick
        ROW_NUMBER() OVER (ORDER BY status_date)
            - ROW_NUMBER() OVER (PARTITION BY status ORDER BY status_date) AS island_id
    FROM market_status
    WHERE status = 'CLOSED'
)
SELECT
    MIN(status_date)        AS streak_start,
    MAX(status_date)        AS streak_end,
    COUNT(*)                AS closed_days_count,
    MAX(status_date) - MIN(status_date) + 1  AS calendar_span_days
FROM closed_days
GROUP BY island_id
ORDER BY streak_start;

/*
How the "island trick" works:
  Row# (global): 1,2,3,4,5 for consecutive closed dates
  Row# (within CLOSED partition): 1,2,3,4,5
  Difference: always the same number for a streak → group them
  When a gap occurs, the global counter jumps but partition counter doesn't → different island_id
*/
```

---

### Q24 — Sessionization: Driver Shift Detection 🔴 Hard

> **Asked at:** LSEG (user session analysis), General Mills (field rep visit analysis).

**Question:** Group consecutive sales rep activities into "sessions" (visits) where the gap between activities is ≤ 30 minutes. Assign a unique session ID per rep.

```sql
WITH time_diffs AS (
    SELECT
        retailer_id,                     -- acting as "rep" or "user"
        sale_date,
        quantity,
        -- Get previous event time for the same rep
        LAG(sale_date) OVER (
            PARTITION BY retailer_id
            ORDER BY sale_date
        )                                AS prev_event_date
    FROM sales
),
break_flags AS (
    SELECT
        retailer_id,
        sale_date,
        -- Flag: 1 = new session begins here (gap > 30 min OR first event)
        CASE
            WHEN prev_event_date IS NULL
              OR (sale_date - prev_event_date) > INTERVAL '30 minutes'
            THEN 1
            ELSE 0
        END                              AS is_new_session
    FROM time_diffs
),
sessions AS (
    SELECT
        retailer_id,
        sale_date,
        -- Cumulative sum of flags = unique session ID per retailer
        SUM(is_new_session) OVER (
            PARTITION BY retailer_id
            ORDER BY sale_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                AS session_id
    FROM break_flags
)
SELECT
    retailer_id,
    session_id,
    MIN(sale_date)       AS session_start,
    MAX(sale_date)       AS session_end,
    COUNT(*)             AS events_in_session
FROM sessions
GROUP BY retailer_id, session_id
ORDER BY retailer_id, session_id;
```

---

## 12. Performance Optimization

### Q25 — Query Optimization: Explain & Index Strategy 🔴 Hard

> **Critical for Senior DS roles. Shows production awareness.**

```sql
-- BEFORE: Slow correlated subquery (runs once per row in outer query)
SELECT
    s.sale_id,
    s.product_id,
    s.quantity * s.unit_price AS revenue,
    -- Correlated subquery: N+1 problem
    (SELECT AVG(s2.quantity * s2.unit_price)
     FROM sales s2
     WHERE s2.product_id = s.product_id) AS avg_product_revenue
FROM sales s;

-- AFTER: Rewrite with window function (single scan)
SELECT
    sale_id,
    product_id,
    quantity * unit_price                              AS revenue,
    AVG(quantity * unit_price) OVER (
        PARTITION BY product_id
    )                                                  AS avg_product_revenue
FROM sales;

-- ─────────────────────────────────────────────────────
-- BEFORE: Double evaluation of LAG in SELECT
SELECT
    ticker,
    trade_date,
    close_price,
    close_price - LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date)
                                                       AS price_change,
    (close_price - LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date))
    / NULLIF(LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date), 0) * 100
                                                       AS pct_change
FROM stock_prices;
-- BAD: Window function evaluated 3 times

-- AFTER: Use CTE to compute once
WITH lagged AS (
    SELECT
        ticker,
        trade_date,
        close_price,
        LAG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date)  AS prev_close
    FROM stock_prices
)
SELECT
    ticker,
    trade_date,
    close_price,
    close_price - prev_close                           AS price_change,
    ROUND((close_price - prev_close)
          / NULLIF(prev_close, 0) * 100, 4)            AS pct_change
FROM lagged;

-- ─────────────────────────────────────────────────────
-- INDEXING STRATEGY (mention in interview)
-- For the sales table, these indexes dramatically speed up common patterns:
CREATE INDEX idx_sales_product_date ON sales(product_id, sale_date);
CREATE INDEX idx_sales_region_date  ON sales(region, sale_date);
CREATE INDEX idx_sales_retailer     ON sales(retailer_id);
-- For stock_prices window functions:
CREATE INDEX idx_stock_ticker_date  ON stock_prices(ticker, trade_date);

-- Use EXPLAIN ANALYZE to verify index usage:
EXPLAIN ANALYZE
SELECT ticker, AVG(close_price)
FROM stock_prices
WHERE ticker = 'RELIANCE'
GROUP BY ticker;
```

**Optimization Checklist for Interviews:**
1. **Avoid correlated subqueries** → rewrite as JOINs or window functions
2. **Filter early** → push `WHERE` conditions as deep as possible
3. **Index join/filter columns**: `(partition_col, order_col)` for window functions
4. **Use CTEs for readability** but be aware modern optimisers inline them (use `MATERIALIZED` hint when needed)
5. **Avoid `SELECT *`** in production — select only needed columns
6. **Use `LIMIT` during development** to avoid full scans on large tables

---

## 13. Python + SQL Integration

### Q26 — SQLAlchemy + pandas: Production-Grade Data Pipeline 🟡 Medium

> **Asked as a system design question at both General Mills and LSEG.**

```python
# ─── Setup & Connection ───────────────────────────────────────────────────────
import pandas as pd
from sqlalchemy import create_engine, text
import logging

# Connection string format: dialect+driver://user:pass@host:port/dbname
# General Mills (Snowflake):
engine_gm = create_engine(
    "snowflake://user:password@account/database/schema"
    "?warehouse=ANALYTICS_WH&role=DS_ROLE"
)

# LSEG (PostgreSQL):
engine_lseg = create_engine(
    "postgresql+psycopg2://user:password@host:5432/market_data",
    pool_size=5,           # Connection pool size
    max_overflow=10,       # Extra connections during peak
    pool_timeout=30        # Seconds to wait for a connection
)


# ─── Q: How do you safely pass user parameters to avoid SQL injection? ───────
def get_sales_by_region(engine, region: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Safe parameterised query — NEVER use f-strings for user inputs."""
    query = text("""
        SELECT
            p.category,
            s.sale_date,
            SUM(s.quantity * s.unit_price) AS revenue
        FROM sales s
        JOIN products p ON s.product_id = p.product_id
        WHERE s.region    = :region
          AND s.sale_date BETWEEN :start_date AND :end_date
        GROUP BY p.category, s.sale_date
        ORDER BY s.sale_date
    """)
    with engine_lseg.connect() as conn:
        df = pd.read_sql(query, conn, params={
            "region": region,
            "start_date": start_date,
            "end_date": end_date
        })
    return df


# ─── Q: How do you handle 100M+ row tables without OOM? ─────────────────────
def process_large_table(engine, chunksize: int = 50_000) -> pd.DataFrame:
    """
    Use chunksize to avoid loading everything into RAM.
    Process each chunk and concatenate filtered results.
    """
    query = "SELECT * FROM massive_sales_table WHERE year = 2024"
    chunks = []
    for chunk in pd.read_sql(query, engine, chunksize=chunksize):
        # Perform lightweight transformation per chunk
        filtered = chunk[chunk["quantity"] > 0].copy()
        filtered["net_revenue"] = filtered["quantity"] * filtered["unit_price"] - filtered["discount_amount"]
        chunks.append(filtered[["sale_id", "product_id", "net_revenue"]])

    return pd.concat(chunks, ignore_index=True)


# ─── Q: Write a processed DataFrame back to SQL ──────────────────────────────
def write_aggregated_results(df: pd.DataFrame, engine, table_name: str) -> None:
    """Write in chunks; use append mode for incremental loads."""
    df.to_sql(
        name=table_name,
        con=engine,
        if_exists="append",   # 'replace' truncates; 'fail' raises error
        index=False,           # Don't write pandas index as a column
        chunksize=10_000,      # Write in batches of 10K rows
        method="multi"         # Use multi-row INSERT for performance
    )
    logging.info(f"Written {len(df)} rows to {table_name}")


# ─── Q: Pandas equivalent of SQL window functions ────────────────────────────
def pandas_window_equivalent(df: pd.DataFrame) -> pd.DataFrame:
    """
    SQL: SELECT ticker, trade_date, close_price,
                AVG(close_price) OVER (PARTITION BY ticker ORDER BY trade_date ROWS BETWEEN 6 PRECEDING AND CURRENT ROW)
         FROM stock_prices
    """
    df = df.sort_values(["ticker", "trade_date"])
    df["moving_avg_7d"] = (
        df.groupby("ticker")["close_price"]
          .transform(lambda x: x.rolling(window=7, min_periods=1).mean())
    )

    # RANK() equivalent
    df["revenue_rank"] = (
        df.groupby("category")["total_revenue"]
          .rank(method="min", ascending=False)  # method='min' = RANK(), 'dense' = DENSE_RANK()
    )

    # LAG() equivalent
    df["prev_month_revenue"] = (
        df.groupby("category")["revenue"]
          .shift(1)                              # shift(1) = LAG(1), shift(-1) = LEAD(1)
    )
    return df


# ─── Q: SQLAlchemy Core — raw SQL execution ──────────────────────────────────
def execute_ddl(engine) -> None:
    """Execute DDL or DML statements (not SELECT)."""
    with engine.begin() as conn:    # begin() = auto-commit on success, rollback on exception
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS ds_outputs (
                run_id      SERIAL PRIMARY KEY,
                model_name  VARCHAR(100),
                run_date    DATE DEFAULT CURRENT_DATE,
                metric_name VARCHAR(100),
                metric_value DECIMAL(12, 4)
            )
        """))
```

---

## 14. Common Interview Gotchas

> [!WARNING]
> These are the most common mistakes that trip up candidates — even experienced ones.

### Gotcha 1: NULL Arithmetic
```sql
-- NULL + anything = NULL
SELECT 100 + NULL;    -- Returns NULL, not 100

-- Always use COALESCE or NULLIF
SELECT COALESCE(discount_amount, 0) + 100 AS adjusted;

-- COUNT(*) vs COUNT(column)
SELECT
    COUNT(*)               AS total_rows,         -- Counts ALL rows including NULLs
    COUNT(discount_amount) AS non_null_discounts;  -- Excludes NULLs
```

### Gotcha 2: Window Function ≠ GROUP BY
```sql
-- WRONG: Can't mix regular column with window function in WHERE
-- Window functions run AFTER WHERE, so this fails:
SELECT *
FROM (SELECT ticker, close_price,
             ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY close_price DESC) AS rn
      FROM stock_prices) sub
WHERE rn = 1;    -- Must filter in outer query or CTE, NOT in same SELECT

-- CORRECT:
WITH ranked AS (
    SELECT ticker, close_price,
           ROW_NUMBER() OVER (PARTITION BY ticker ORDER BY close_price DESC) AS rn
    FROM stock_prices
)
SELECT * FROM ranked WHERE rn = 1;
```

### Gotcha 3: RANK vs DENSE_RANK in Top-N Filtering
```sql
-- If two products tie at rank 2, RANK() gives 1,2,2,4 → "top 3" with rnk<=3 misses rank 4
-- Use DENSE_RANK() for "top N items": 1,2,2,3 → rnk<=3 catches everything through revenue level 3
-- Use ROW_NUMBER() when you need EXACTLY N rows (no ties)
```

### Gotcha 4: Date vs Timestamp Comparison
```sql
-- Pitfall: comparing DATE with TIMESTAMP truncates incorrectly
WHERE sale_date = '2024-01-15'          -- OK if sale_date is DATE type
WHERE sale_time = '2024-01-15'          -- WRONG if sale_time is TIMESTAMP (misses times ≠ midnight)

-- Correct for TIMESTAMP:
WHERE sale_time >= '2024-01-15 00:00:00'
  AND sale_time <  '2024-01-16 00:00:00'
-- OR:
WHERE DATE(sale_time) = '2024-01-15'   -- Slower; prevents index use
```

### Gotcha 5: HAVING with Non-Aggregate Columns
```sql
-- WRONG: region is not aggregated — this is a logic error
SELECT region, COUNT(*) FROM sales GROUP BY category HAVING region = 'South';

-- CORRECT: Filter non-aggregated columns in WHERE
SELECT region, COUNT(*) FROM sales WHERE region = 'South' GROUP BY region;
```

### Gotcha 6: Self-Join Duplicate Pairs
```sql
-- Pairs problem: (A,B) and (B,A) are the same pair
-- Prevent with: a.id < b.id  (always ensures canonical ordering)
JOIN trades b ON a.account_id = b.account_id AND a.trade_id < b.trade_id
```

### Gotcha 7: Division by Zero
```sql
-- Always use NULLIF to prevent division by zero
-- NULLIF(x, 0) returns NULL when x=0, preventing runtime error
ROUND(revenue / NULLIF(cost, 0), 2)   AS margin_ratio
```

### Gotcha 8: DISTINCT in Window Functions
```sql
-- COUNT(DISTINCT col) OVER (...) is NOT supported in most SQL dialects
-- Workaround: use a subquery or approximate count
-- For rolling distinct users, use self-join with date range (see Q4 research notes)
```

---

## 🎯 Quick Topic Checklist

| Topic | Covered | Difficulty |
|-------|---------|-----------|
| WHERE vs HAVING | ✅ Q1 | 🟢 Easy |
| JOIN types + anti-join | ✅ Q2 | 🟢 Easy |
| RANK / DENSE_RANK / ROW_NUMBER | ✅ Q3 | 🟡 Medium |
| Moving Average (window) | ✅ Q4 | 🟡 Medium |
| LAG / LEAD (MoM growth) | ✅ Q5 | 🟡 Medium |
| Top-N per group | ✅ Q6 | 🟡 Medium |
| Cumulative share / Pareto | ✅ Q7 | 🔴 Hard |
| NTILE bucketing | ✅ Q8 | 🟡 Medium |
| Recursive CTE hierarchy | ✅ Q9 | 🔴 Hard |
| Cohort / Retention analysis | ✅ Q10 | 🔴 Hard |
| Self-join time window | ✅ Q11 | 🔴 Hard |
| Multi-table join dashboard | ✅ Q12 | 🟡 Medium |
| Pivot with CASE WHEN | ✅ Q13 | 🟡 Medium |
| Deduplication with ROW_NUMBER | ✅ Q14 | 🟡 Medium |
| Date spine / fill missing | ✅ Q15 | 🔴 Hard |
| Date arithmetic / age | ✅ Q16 | 🟢 Easy |
| String parsing / cleaning | ✅ Q17 | 🟡 Medium |
| Inventory / DOI (FMCG) | ✅ Q18 | 🟡 Medium |
| Trade Promotion ROI (FMCG) | ✅ Q19 | 🔴 Hard |
| Bollinger Bands (Finance) | ✅ Q20 | 🔴 Hard |
| Daily returns / Volatility | ✅ Q21 | 🔴 Hard |
| RFM Segmentation | ✅ Q22 | 🔴 Hard |
| Gaps & Islands (streaks) | ✅ Q23 | 🔴 Hard |
| Sessionization | ✅ Q24 | 🔴 Hard |
| Query optimization | ✅ Q25 | 🔴 Hard |
| Python + SQLAlchemy + pandas | ✅ Q26 | 🟡 Medium |
| Common Gotchas | ✅ Sec 14 | All levels |

---

## 📚 Recommended Practice Resources

| Resource | Focus | URL |
|----------|-------|-----|
| DataLemur | SQL for DS interviews | datalemur.com |
| StrataScatch | Real company SQL questions | stratascratch.com |
| LeetCode SQL | Algorithmic SQL | leetcode.com/study-plan/sql |
| Mode Analytics SQL Tutorial | Analytics SQL | mode.com/sql-tutorial |
| InterviewQuery | General Mills / LSEG style | interviewquery.com |

---

*Last Updated: November 2026 | Prepared for Senior DS interviews at General Mills India & LSEG India*

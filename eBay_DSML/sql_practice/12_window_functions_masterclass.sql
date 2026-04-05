-- ================================================================
-- eBay DS/ML — Advanced SQL Round 2: Window Functions Masterclass
-- ================================================================
-- 15 problems focused EXCLUSIVELY on window functions.
-- These are the #1 most tested SQL concept at eBay.
-- Uses the same schema from 01_marketplace_queries.sql
-- ================================================================

-- ================================================================
-- PROBLEM 1: Rank Sellers Within Each Category ⭐⭐
-- ================================================================
-- For each category, rank sellers by total GMV.
-- Show seller_id, category, total_gmv, rank, dense_rank.
-- Explain the difference between RANK and DENSE_RANK.

-- ANSWER:
/*
SELECT 
    l.category,
    t.seller_id,
    SUM(t.sale_price * t.quantity) AS total_gmv,
    RANK() OVER (PARTITION BY l.category ORDER BY SUM(t.sale_price * t.quantity) DESC) AS gmv_rank,
    DENSE_RANK() OVER (PARTITION BY l.category ORDER BY SUM(t.sale_price * t.quantity) DESC) AS gmv_dense_rank
FROM transactions t
JOIN listings l ON t.listing_id = l.listing_id
WHERE t.status = 'completed'
GROUP BY l.category, t.seller_id
ORDER BY l.category, gmv_rank;

-- RANK: Skips numbers after ties (1, 2, 2, 4)
-- DENSE_RANK: Does not skip (1, 2, 2, 3)
-- ROW_NUMBER: Always unique (1, 2, 3, 4) — arbitrary for ties
*/


-- ================================================================
-- PROBLEM 2: Running Total of GMV ⭐⭐
-- ================================================================
-- Calculate a daily running total (cumulative sum) of GMV.

-- ANSWER:
/*
SELECT 
    transaction_date::date AS day,
    SUM(sale_price * quantity) AS daily_gmv,
    SUM(SUM(sale_price * quantity)) OVER (
        ORDER BY transaction_date::date
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
    ) AS running_total_gmv
FROM transactions
WHERE status = 'completed'
GROUP BY transaction_date::date
ORDER BY day;
*/


-- ================================================================
-- PROBLEM 3: Percentage of Category Total ⭐⭐
-- ================================================================
-- For each listing, show its sale price as a % of its category total.

-- ANSWER:
/*
SELECT 
    t.listing_id,
    l.category,
    t.sale_price,
    SUM(t.sale_price) OVER (PARTITION BY l.category) AS category_total,
    ROUND(t.sale_price * 100.0 / SUM(t.sale_price) OVER (PARTITION BY l.category), 4) AS pct_of_category
FROM transactions t
JOIN listings l ON t.listing_id = l.listing_id
WHERE t.status = 'completed';
*/


-- ================================================================
-- PROBLEM 4: LAG/LEAD — Day-over-Day Revenue Change ⭐⭐
-- ================================================================
-- Show daily revenue with previous day's revenue and % change.

-- ANSWER:
/*
WITH daily_rev AS (
    SELECT 
        transaction_date::date AS day,
        SUM(sale_price * quantity) AS revenue
    FROM transactions
    WHERE status = 'completed'
    GROUP BY 1
)
SELECT 
    day,
    revenue,
    LAG(revenue, 1) OVER (ORDER BY day) AS prev_day_revenue,
    LEAD(revenue, 1) OVER (ORDER BY day) AS next_day_revenue,
    ROUND((revenue - LAG(revenue) OVER (ORDER BY day)) * 100.0 
          / NULLIF(LAG(revenue) OVER (ORDER BY day), 0), 2) AS dod_change_pct
FROM daily_rev;
*/


-- ================================================================
-- PROBLEM 5: FIRST_VALUE / LAST_VALUE ⭐⭐
-- ================================================================
-- For each user, show their first and last purchase category.

-- ANSWER:
/*
SELECT DISTINCT
    buyer_id,
    FIRST_VALUE(l.category) OVER (
        PARTITION BY buyer_id ORDER BY transaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS first_purchase_category,
    LAST_VALUE(l.category) OVER (
        PARTITION BY buyer_id ORDER BY transaction_date
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS last_purchase_category
FROM transactions t
JOIN listings l ON t.listing_id = l.listing_id
WHERE t.status = 'completed';
*/


-- ================================================================
-- PROBLEM 6: NTH_VALUE — 2nd Highest Transaction ⭐⭐
-- ================================================================
-- For each buyer, find the value of their 2nd highest transaction.

-- ANSWER:
/*
SELECT DISTINCT
    buyer_id,
    NTH_VALUE(sale_price, 2) OVER (
        PARTITION BY buyer_id 
        ORDER BY sale_price DESC
        ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING
    ) AS second_highest_purchase
FROM transactions
WHERE status = 'completed';
*/


-- ================================================================
-- PROBLEM 7: NTILE — Quartile Bucketing ⭐⭐
-- ================================================================
-- Divide all items into price quartiles within each category.
-- Show listing_id, price, quartile.

-- ANSWER:
/*
SELECT 
    listing_id,
    category,
    price,
    NTILE(4) OVER (PARTITION BY category ORDER BY price) AS price_quartile
FROM listings
WHERE status = 'active';
*/


-- ================================================================
-- PROBLEM 8: Moving Average with ROWS vs RANGE ⭐⭐⭐
-- ================================================================
-- Calculate 7-day moving average of GMV.
-- Explain ROWS vs RANGE frame specification.

-- ANSWER:
/*
WITH daily_gmv AS (
    SELECT transaction_date::date AS day, SUM(sale_price * quantity) AS gmv
    FROM transactions WHERE status = 'completed'
    GROUP BY 1
)
SELECT 
    day, gmv,
    -- ROWS: exactly the previous 6 rows + current (physical window)
    AVG(gmv) OVER (ORDER BY day ROWS BETWEEN 6 PRECEDING AND CURRENT ROW) AS moving_avg_rows,
    -- RANGE: all rows within 6 days of current date (logical window)
    AVG(gmv) OVER (ORDER BY day RANGE BETWEEN INTERVAL '6 days' PRECEDING AND CURRENT ROW) AS moving_avg_range
FROM daily_gmv;

-- DIFFERENCE:
-- ROWS: counts rows regardless of gaps in dates
-- RANGE: accounts for missing dates (weekends, holidays)
-- If there are no missing dates, they produce the same result
*/


-- ================================================================
-- PROBLEM 9: Gap Detection — Days Between Purchases ⭐⭐⭐
-- ================================================================
-- For each buyer, calculate the days between consecutive purchases.
-- Flag users with a gap > 60 days as "at risk of churn".

-- ANSWER:
/*
WITH purchase_gaps AS (
    SELECT 
        buyer_id,
        transaction_date::date AS purchase_date,
        LAG(transaction_date::date) OVER (
            PARTITION BY buyer_id ORDER BY transaction_date
        ) AS prev_purchase_date,
        transaction_date::date - LAG(transaction_date::date) OVER (
            PARTITION BY buyer_id ORDER BY transaction_date
        ) AS days_between
    FROM transactions
    WHERE status = 'completed'
)
SELECT 
    buyer_id,
    AVG(days_between) AS avg_days_between_purchases,
    MAX(days_between) AS max_gap_days,
    CASE WHEN MAX(days_between) > 60 THEN 'AT_RISK' ELSE 'HEALTHY' END AS churn_risk
FROM purchase_gaps
WHERE days_between IS NOT NULL
GROUP BY buyer_id
ORDER BY max_gap_days DESC;
*/


-- ================================================================
-- PROBLEM 10: Year-over-Year Growth per Category ⭐⭐⭐
-- ================================================================
-- Calculate YoY GMV growth for each category.

-- ANSWER:
/*
WITH yearly_gmv AS (
    SELECT 
        l.category,
        EXTRACT(YEAR FROM t.transaction_date) AS year,
        SUM(t.sale_price * t.quantity) AS gmv
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
    GROUP BY l.category, EXTRACT(YEAR FROM t.transaction_date)
)
SELECT 
    category, year, gmv,
    LAG(gmv) OVER (PARTITION BY category ORDER BY year) AS prev_year_gmv,
    ROUND((gmv - LAG(gmv) OVER (PARTITION BY category ORDER BY year)) * 100.0 
          / NULLIF(LAG(gmv) OVER (PARTITION BY category ORDER BY year), 0), 2) AS yoy_growth_pct
FROM yearly_gmv
ORDER BY category, year;
*/


-- ================================================================
-- PROBLEM 11: Median Calculation Using Window Functions ⭐⭐⭐
-- ================================================================
-- Calculate the median sale price per category.
-- (PERCENTILE_CONT is a window/aggregate function)

-- ANSWER:
/*
SELECT DISTINCT
    l.category,
    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY t.sale_price) 
        AS median_sale_price
FROM transactions t
JOIN listings l ON t.listing_id = l.listing_id
WHERE t.status = 'completed'
GROUP BY l.category
ORDER BY median_sale_price DESC;
*/


-- ================================================================
-- PROBLEM 12: Top Transaction Per User (No Subquery) ⭐⭐
-- ================================================================
-- Get each user's highest-value transaction using window functions.
-- Do NOT use a subquery or CTE.

-- ANSWER:
/*
SELECT * FROM (
    SELECT 
        buyer_id,
        transaction_id,
        sale_price,
        ROW_NUMBER() OVER (PARTITION BY buyer_id ORDER BY sale_price DESC) AS rn
    FROM transactions
    WHERE status = 'completed'
) ranked
WHERE rn = 1;
*/


-- ================================================================
-- PROBLEM 13: Consecutive Days Active ⭐⭐⭐
-- ================================================================
-- Find the longest streak of consecutive days each user was active
-- (had at least one transaction).

-- ANSWER:
/*
WITH active_days AS (
    SELECT DISTINCT buyer_id, transaction_date::date AS active_date
    FROM transactions WHERE status = 'completed'
),
with_gaps AS (
    SELECT *,
        active_date - ROW_NUMBER() OVER (
            PARTITION BY buyer_id ORDER BY active_date
        )::int AS grp
    FROM active_days
),
streaks AS (
    SELECT buyer_id, grp,
        COUNT(*) AS streak_length,
        MIN(active_date) AS streak_start,
        MAX(active_date) AS streak_end
    FROM with_gaps
    GROUP BY buyer_id, grp
)
SELECT buyer_id, MAX(streak_length) AS longest_streak
FROM streaks
GROUP BY buyer_id
ORDER BY longest_streak DESC
LIMIT 20;
*/


-- ================================================================
-- PROBLEM 14: Market Share by Category Over Time ⭐⭐⭐
-- ================================================================
-- For each month + category, calculate each seller's % share of
-- category GMV. Only show sellers with >5% share.

-- ANSWER:
/*
WITH monthly_seller_gmv AS (
    SELECT 
        DATE_TRUNC('month', t.transaction_date) AS month,
        l.category,
        t.seller_id,
        SUM(t.sale_price * t.quantity) AS seller_gmv
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
    GROUP BY 1, 2, 3
)
SELECT 
    month, category, seller_id, seller_gmv,
    SUM(seller_gmv) OVER (PARTITION BY month, category) AS category_total_gmv,
    ROUND(seller_gmv * 100.0 / SUM(seller_gmv) OVER (PARTITION BY month, category), 2) AS market_share_pct
FROM monthly_seller_gmv
HAVING ROUND(seller_gmv * 100.0 / SUM(seller_gmv) OVER (PARTITION BY month, category), 2) > 5
ORDER BY month, category, market_share_pct DESC;
*/


-- ================================================================
-- PROBLEM 15: Window Frames Comparison ⭐⭐⭐⭐
-- ================================================================
-- Show the difference between these frame specifications:
-- ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW  (running total)
-- ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING          (centered window)
-- ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING   (reverse running)

-- ANSWER:
/*
WITH daily AS (
    SELECT transaction_date::date AS day, SUM(sale_price) AS rev
    FROM transactions WHERE status = 'completed'
    GROUP BY 1
)
SELECT 
    day, rev,
    -- Running total from start to current row
    SUM(rev) OVER (ORDER BY day ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
    
    -- Centered 5-day window (2 before + current + 2 after)
    AVG(rev) OVER (ORDER BY day ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS centered_5d_avg,
    
    -- Reverse running total (current row to end)
    SUM(rev) OVER (ORDER BY day ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING) AS reverse_running
FROM daily
ORDER BY day;
*/

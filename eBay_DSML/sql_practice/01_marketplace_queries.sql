-- ================================================================
-- eBay DS/ML Interview — SQL Practice Problems
-- ================================================================
-- These problems simulate eBay's marketplace data model.
-- Practice writing solutions BEFORE looking at the answers below.
-- 
-- Difficulty: ⭐ Easy  ⭐⭐ Medium  ⭐⭐⭐ Hard  ⭐⭐⭐⭐ Expert
-- ================================================================

-- ============================================================
-- SCHEMA DEFINITION (Use this to set up a local practice DB)
-- ============================================================

-- Run against: PostgreSQL, MySQL 8+, or SQLite (some syntax may vary)

CREATE TABLE users (
    user_id         INT PRIMARY KEY,
    username        VARCHAR(100),
    email           VARCHAR(200),
    country         VARCHAR(50),
    signup_date     DATE,
    user_type       VARCHAR(10)   -- 'buyer', 'seller', 'both'
);

CREATE TABLE listings (
    listing_id      INT PRIMARY KEY,
    seller_id       INT REFERENCES users(user_id),
    title           VARCHAR(500),
    category        VARCHAR(100),
    sub_category    VARCHAR(100),
    price           DECIMAL(10,2),
    condition       VARCHAR(20),    -- 'New', 'Like New', 'Good', 'Fair', 'Poor'
    listing_date    DATE,
    status          VARCHAR(20),    -- 'active', 'sold', 'ended', 'removed'
    listing_type    VARCHAR(20)     -- 'auction', 'buy_it_now', 'both'
);

CREATE TABLE transactions (
    transaction_id  INT PRIMARY KEY,
    buyer_id        INT REFERENCES users(user_id),
    seller_id       INT REFERENCES users(user_id),
    listing_id      INT REFERENCES listings(listing_id),
    transaction_date TIMESTAMP,
    sale_price      DECIMAL(10,2),
    quantity        INT DEFAULT 1,
    shipping_cost   DECIMAL(10,2),
    status          VARCHAR(20),    -- 'completed', 'cancelled', 'returned', 'pending'
    payment_method  VARCHAR(50),
    country         VARCHAR(50)
);

CREATE TABLE search_logs (
    search_id       INT PRIMARY KEY,
    user_id         INT REFERENCES users(user_id),
    query           VARCHAR(500),
    search_time     TIMESTAMP,
    num_results     INT,
    category_filter VARCHAR(100)
);

CREATE TABLE click_events (
    click_id        INT PRIMARY KEY,
    search_id       INT REFERENCES search_logs(search_id),
    user_id         INT REFERENCES users(user_id),
    listing_id      INT REFERENCES listings(listing_id),
    click_time      TIMESTAMP,
    position        INT,            -- position in search results (1-based)
    page_number     INT
);

CREATE TABLE page_views (
    view_id         INT PRIMARY KEY,
    user_id         INT REFERENCES users(user_id),
    listing_id      INT REFERENCES listings(listing_id),
    view_time       TIMESTAMP,
    source          VARCHAR(50),    -- 'search', 'recommendation', 'direct', 'email'
    device_type     VARCHAR(20)     -- 'mobile', 'desktop', 'tablet'
);

CREATE TABLE cart_events (
    cart_id         INT PRIMARY KEY,
    user_id         INT REFERENCES users(user_id),
    listing_id      INT REFERENCES listings(listing_id),
    event_type      VARCHAR(20),    -- 'add', 'remove'
    event_time      TIMESTAMP
);

CREATE TABLE seller_reviews (
    review_id       INT PRIMARY KEY,
    seller_id       INT REFERENCES users(user_id),
    buyer_id        INT REFERENCES users(user_id),
    transaction_id  INT REFERENCES transactions(transaction_id),
    rating          INT CHECK (rating BETWEEN 1 AND 5),
    review_text     TEXT,
    review_date     DATE
);


-- ================================================================
-- PROBLEM 1: Total GMV by Category (Last 90 Days) ⭐
-- ================================================================
-- Calculate total GMV (sale_price * quantity) and unique buyer count 
-- by product category for the last 90 days.
-- Exclude cancelled and returned orders.

-- YOUR SOLUTION:


-- ANSWER:
/*
SELECT 
    l.category,
    SUM(t.sale_price * t.quantity) AS total_gmv,
    COUNT(DISTINCT t.buyer_id) AS unique_buyers
FROM transactions t
JOIN listings l ON t.listing_id = l.listing_id
WHERE t.status = 'completed'
  AND t.transaction_date >= CURRENT_DATE - INTERVAL '90 days'
GROUP BY l.category
ORDER BY total_gmv DESC;
*/


-- ================================================================
-- PROBLEM 2: Month-over-Month GMV Growth ⭐⭐
-- ================================================================
-- Calculate the month-over-month percentage change in GMV for 2025.
-- Use window functions (LAG).

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH monthly_gmv AS (
    SELECT 
        DATE_TRUNC('month', transaction_date) AS month,
        SUM(sale_price * quantity) AS gmv
    FROM transactions
    WHERE status = 'completed'
      AND EXTRACT(YEAR FROM transaction_date) = 2025
    GROUP BY 1
)
SELECT 
    month,
    gmv,
    LAG(gmv) OVER (ORDER BY month) AS prev_month_gmv,
    ROUND(
        (gmv - LAG(gmv) OVER (ORDER BY month)) * 100.0 
        / NULLIF(LAG(gmv) OVER (ORDER BY month), 0), 
        2
    ) AS mom_growth_pct
FROM monthly_gmv
ORDER BY month;
*/


-- ================================================================
-- PROBLEM 3: Top 5 Sellers by GMV per Region ⭐⭐
-- ================================================================
-- Find the top 5 sellers by total GMV for each country/region.
-- Include seller_id, country, total_gmv, and rank.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH seller_gmv AS (
    SELECT 
        t.seller_id,
        t.country,
        SUM(t.sale_price * t.quantity) AS total_gmv
    FROM transactions t
    WHERE t.status = 'completed'
    GROUP BY t.seller_id, t.country
),
ranked AS (
    SELECT *,
        DENSE_RANK() OVER (
            PARTITION BY country 
            ORDER BY total_gmv DESC
        ) AS gmv_rank
    FROM seller_gmv
)
SELECT seller_id, country, total_gmv, gmv_rank
FROM ranked
WHERE gmv_rank <= 5
ORDER BY country, gmv_rank;
*/


-- ================================================================
-- PROBLEM 4: Conversion Funnel Analysis ⭐⭐
-- ================================================================
-- Calculate the conversion funnel by category:
--   Step 1: Page Views
--   Step 2: Add to Cart
--   Step 3: Purchase
-- Show view_to_cart_rate and cart_to_purchase_rate.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH views AS (
    SELECT l.category, COUNT(DISTINCT pv.user_id || '-' || pv.listing_id) AS view_count
    FROM page_views pv
    JOIN listings l ON pv.listing_id = l.listing_id
    GROUP BY l.category
),
carts AS (
    SELECT l.category, COUNT(DISTINCT ce.user_id || '-' || ce.listing_id) AS cart_count
    FROM cart_events ce
    JOIN listings l ON ce.listing_id = l.listing_id
    WHERE ce.event_type = 'add'
    GROUP BY l.category
),
purchases AS (
    SELECT l.category, COUNT(DISTINCT t.buyer_id || '-' || t.listing_id) AS purchase_count
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
    GROUP BY l.category
)
SELECT 
    v.category,
    v.view_count,
    c.cart_count,
    p.purchase_count,
    ROUND(c.cart_count * 100.0 / NULLIF(v.view_count, 0), 2) AS view_to_cart_pct,
    ROUND(p.purchase_count * 100.0 / NULLIF(c.cart_count, 0), 2) AS cart_to_purchase_pct
FROM views v
LEFT JOIN carts c ON v.category = c.category
LEFT JOIN purchases p ON v.category = p.category
ORDER BY v.view_count DESC;
*/


-- ================================================================
-- PROBLEM 5: First vs Latest Purchase Comparison ⭐⭐
-- ================================================================
-- For each user, find their first purchase date/amount and their
-- most recent purchase date/amount. Calculate the % change.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH user_orders AS (
    SELECT 
        buyer_id,
        sale_price,
        transaction_date,
        ROW_NUMBER() OVER (PARTITION BY buyer_id ORDER BY transaction_date ASC) AS rn_first,
        ROW_NUMBER() OVER (PARTITION BY buyer_id ORDER BY transaction_date DESC) AS rn_latest
    FROM transactions
    WHERE status = 'completed'
)
SELECT 
    f.buyer_id,
    f.transaction_date AS first_purchase_date,
    f.sale_price AS first_order_value,
    l.transaction_date AS latest_purchase_date,
    l.sale_price AS latest_order_value,
    ROUND((l.sale_price - f.sale_price) * 100.0 / NULLIF(f.sale_price, 0), 2) AS pct_change
FROM user_orders f
JOIN user_orders l ON f.buyer_id = l.buyer_id
WHERE f.rn_first = 1 AND l.rn_latest = 1
ORDER BY pct_change DESC;
*/


-- ================================================================
-- PROBLEM 6: 3-Month Rolling Average of Active Buyers ⭐⭐
-- ================================================================
-- Calculate a 3-month rolling average of monthly active buyers.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH monthly_buyers AS (
    SELECT 
        DATE_TRUNC('month', transaction_date) AS month,
        COUNT(DISTINCT buyer_id) AS active_buyers
    FROM transactions
    WHERE status = 'completed'
    GROUP BY 1
)
SELECT 
    month,
    active_buyers,
    ROUND(AVG(active_buyers) OVER (
        ORDER BY month 
        ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
    ), 0) AS rolling_3mo_avg
FROM monthly_buyers
ORDER BY month;
*/


-- ================================================================
-- PROBLEM 7: Duplicate Listing Detection ⭐⭐
-- ================================================================
-- Identify duplicate listings: same seller, same title, same 
-- category, listed within a 7-day window.

-- YOUR SOLUTION:


-- ANSWER:
/*
SELECT 
    a.listing_id AS original_listing_id,
    b.listing_id AS duplicate_listing_id,
    a.seller_id,
    a.title,
    a.category,
    a.listing_date AS original_date,
    b.listing_date AS duplicate_date
FROM listings a
JOIN listings b 
    ON a.seller_id = b.seller_id
    AND a.title = b.title
    AND a.category = b.category
    AND a.listing_id < b.listing_id
    AND ABS(b.listing_date - a.listing_date) <= 7
ORDER BY a.seller_id, a.listing_date;
*/


-- ================================================================
-- PROBLEM 8: Monthly Retention Cohort ⭐⭐⭐
-- ================================================================
-- Build a retention cohort table:
-- For users who first purchased in month M, what % return in
-- months M+1, M+2, ..., M+6?

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH first_purchase AS (
    SELECT 
        buyer_id,
        DATE_TRUNC('month', MIN(transaction_date)) AS cohort_month
    FROM transactions
    WHERE status = 'completed'
    GROUP BY buyer_id
),
activity AS (
    SELECT 
        t.buyer_id,
        fp.cohort_month,
        DATE_TRUNC('month', t.transaction_date) AS activity_month,
        (EXTRACT(YEAR FROM DATE_TRUNC('month', t.transaction_date)) * 12 
         + EXTRACT(MONTH FROM DATE_TRUNC('month', t.transaction_date)))
        - (EXTRACT(YEAR FROM fp.cohort_month) * 12 
         + EXTRACT(MONTH FROM fp.cohort_month)) AS months_since_first
    FROM transactions t
    JOIN first_purchase fp ON t.buyer_id = fp.buyer_id
    WHERE t.status = 'completed'
),
cohort_size AS (
    SELECT cohort_month, COUNT(DISTINCT buyer_id) AS cohort_users
    FROM first_purchase
    GROUP BY cohort_month
)
SELECT 
    a.cohort_month,
    cs.cohort_users,
    a.months_since_first,
    COUNT(DISTINCT a.buyer_id) AS retained_users,
    ROUND(COUNT(DISTINCT a.buyer_id) * 100.0 / cs.cohort_users, 2) AS retention_pct
FROM activity a
JOIN cohort_size cs ON a.cohort_month = cs.cohort_month
WHERE a.months_since_first BETWEEN 0 AND 6
GROUP BY a.cohort_month, cs.cohort_users, a.months_since_first
ORDER BY a.cohort_month, a.months_since_first;
*/


-- ================================================================
-- PROBLEM 9: Session Definition & Analysis ⭐⭐⭐
-- ================================================================
-- Define a session as a sequence of click events where no two 
-- consecutive events are more than 30 minutes apart.
-- Find: avg session duration, avg events per session.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH ordered_clicks AS (
    SELECT 
        user_id,
        click_time,
        LAG(click_time) OVER (PARTITION BY user_id ORDER BY click_time) AS prev_click_time
    FROM click_events
),
session_boundaries AS (
    SELECT *,
        CASE 
            WHEN prev_click_time IS NULL 
              OR EXTRACT(EPOCH FROM (click_time - prev_click_time)) > 1800 
            THEN 1 
            ELSE 0 
        END AS is_new_session
    FROM ordered_clicks
),
session_ids AS (
    SELECT *,
        SUM(is_new_session) OVER (
            PARTITION BY user_id ORDER BY click_time
        ) AS session_id
    FROM session_boundaries
),
session_stats AS (
    SELECT 
        user_id,
        session_id,
        COUNT(*) AS events_in_session,
        MIN(click_time) AS session_start,
        MAX(click_time) AS session_end,
        EXTRACT(EPOCH FROM (MAX(click_time) - MIN(click_time))) / 60.0 AS session_duration_min
    FROM session_ids
    GROUP BY user_id, session_id
)
SELECT 
    ROUND(AVG(session_duration_min), 2) AS avg_session_duration_min,
    ROUND(AVG(events_in_session), 2) AS avg_events_per_session,
    COUNT(*) AS total_sessions
FROM session_stats;
*/


-- ================================================================
-- PROBLEM 10: Search-to-Purchase Attribution ⭐⭐⭐
-- ================================================================
-- Attribute purchases to the search click that led to them 
-- (within a 24-hour window). Calculate revenue per search query.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH attributed AS (
    SELECT 
        sl.query,
        ce.click_id,
        ce.listing_id,
        ce.click_time,
        t.transaction_id,
        t.sale_price * t.quantity AS revenue,
        ROW_NUMBER() OVER (
            PARTITION BY t.transaction_id 
            ORDER BY ce.click_time DESC
        ) AS rn  -- last-click attribution
    FROM click_events ce
    JOIN search_logs sl ON ce.search_id = sl.search_id
    JOIN transactions t 
        ON ce.listing_id = t.listing_id
        AND ce.user_id = t.buyer_id
        AND t.transaction_date BETWEEN ce.click_time 
            AND ce.click_time + INTERVAL '24 hours'
    WHERE t.status = 'completed'
)
SELECT 
    query,
    COUNT(DISTINCT transaction_id) AS attributed_purchases,
    SUM(revenue) AS total_attributed_revenue,
    ROUND(SUM(revenue) / NULLIF(COUNT(DISTINCT transaction_id), 0), 2) AS avg_revenue_per_purchase
FROM attributed
WHERE rn = 1
GROUP BY query
ORDER BY total_attributed_revenue DESC
LIMIT 20;
*/


-- ================================================================
-- PROBLEM 11: Seller Churn Feature Table ⭐⭐⭐
-- ================================================================
-- Create a feature table for seller churn prediction:
-- For each seller: GMV in last 30/60/90 days, # active listings,
-- avg time to sell, # buyer complaints (ratings <= 2).

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH gmv_windows AS (
    SELECT 
        seller_id,
        SUM(CASE WHEN transaction_date >= CURRENT_DATE - INTERVAL '30 days' 
            THEN sale_price * quantity ELSE 0 END) AS gmv_30d,
        SUM(CASE WHEN transaction_date >= CURRENT_DATE - INTERVAL '60 days' 
            THEN sale_price * quantity ELSE 0 END) AS gmv_60d,
        SUM(CASE WHEN transaction_date >= CURRENT_DATE - INTERVAL '90 days' 
            THEN sale_price * quantity ELSE 0 END) AS gmv_90d
    FROM transactions
    WHERE status = 'completed'
    GROUP BY seller_id
),
active_listings AS (
    SELECT seller_id, COUNT(*) AS num_active_listings
    FROM listings
    WHERE status = 'active'
    GROUP BY seller_id
),
time_to_sell AS (
    SELECT 
        l.seller_id,
        AVG(t.transaction_date::date - l.listing_date) AS avg_days_to_sell
    FROM listings l
    JOIN transactions t ON l.listing_id = t.listing_id
    WHERE t.status = 'completed' AND l.status = 'sold'
    GROUP BY l.seller_id
),
complaints AS (
    SELECT seller_id, COUNT(*) AS num_complaints
    FROM seller_reviews
    WHERE rating <= 2
    GROUP BY seller_id
)
SELECT 
    u.user_id AS seller_id,
    COALESCE(g.gmv_30d, 0) AS gmv_30d,
    COALESCE(g.gmv_60d, 0) AS gmv_60d,
    COALESCE(g.gmv_90d, 0) AS gmv_90d,
    COALESCE(al.num_active_listings, 0) AS num_active_listings,
    COALESCE(tts.avg_days_to_sell, 0) AS avg_days_to_sell,
    COALESCE(c.num_complaints, 0) AS num_complaints
FROM users u
LEFT JOIN gmv_windows g ON u.user_id = g.seller_id
LEFT JOIN active_listings al ON u.user_id = al.seller_id
LEFT JOIN time_to_sell tts ON u.user_id = tts.seller_id
LEFT JOIN complaints c ON u.user_id = c.seller_id
WHERE u.user_type IN ('seller', 'both');
*/


-- ================================================================
-- PROBLEM 12: Quarter-over-Quarter Trend Detection ⭐⭐⭐
-- ================================================================
-- Identify categories where GMV grew by more than 50% 
-- quarter-over-quarter for at least 2 consecutive quarters.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH quarterly_gmv AS (
    SELECT 
        l.category,
        DATE_TRUNC('quarter', t.transaction_date) AS quarter,
        SUM(t.sale_price * t.quantity) AS gmv
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
    GROUP BY l.category, DATE_TRUNC('quarter', t.transaction_date)
),
with_growth AS (
    SELECT *,
        LAG(gmv) OVER (PARTITION BY category ORDER BY quarter) AS prev_quarter_gmv,
        CASE 
            WHEN LAG(gmv) OVER (PARTITION BY category ORDER BY quarter) > 0
            THEN (gmv - LAG(gmv) OVER (PARTITION BY category ORDER BY quarter)) 
                 / LAG(gmv) OVER (PARTITION BY category ORDER BY quarter)
            ELSE NULL 
        END AS qoq_growth
    FROM quarterly_gmv
),
consecutive_growth AS (
    SELECT *,
        CASE WHEN qoq_growth > 0.5 THEN 1 ELSE 0 END AS above_50,
        CASE WHEN LAG(qoq_growth) OVER (PARTITION BY category ORDER BY quarter) > 0.5 
             AND qoq_growth > 0.5 THEN 1 ELSE 0 END AS consecutive_above_50
    FROM with_growth
)
SELECT DISTINCT category
FROM consecutive_growth
WHERE consecutive_above_50 = 1;
*/


-- ================================================================
-- PROBLEM 13: Price Percentile Filtering ⭐⭐
-- ================================================================
-- Select active listings where the price is between the 25th 
-- and 75th percentile of that category's price distribution.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH percentiles AS (
    SELECT 
        listing_id, category, price,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY price) 
            OVER (PARTITION BY category) AS p25,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY price) 
            OVER (PARTITION BY category) AS p75
    FROM listings
    WHERE status = 'active'
)
SELECT listing_id, category, price, p25, p75
FROM percentiles
WHERE price BETWEEN p25 AND p75
ORDER BY category, price;
*/


-- ================================================================
-- PROBLEM 14: Supply-Demand Imbalance ⭐⭐
-- ================================================================
-- Find categories that are oversupplied (listings:purchases > 100:1) 
-- or undersupplied (< 2:1) in the last 30 days.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH supply AS (
    SELECT category, COUNT(*) AS active_listings
    FROM listings
    WHERE status = 'active'
    GROUP BY category
),
demand AS (
    SELECT l.category, COUNT(*) AS purchases_30d
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
      AND t.transaction_date >= CURRENT_DATE - INTERVAL '30 days'
    GROUP BY l.category
)
SELECT 
    s.category,
    s.active_listings,
    COALESCE(d.purchases_30d, 0) AS purchases_30d,
    ROUND(s.active_listings * 1.0 / NULLIF(d.purchases_30d, 0), 1) AS supply_demand_ratio,
    CASE 
        WHEN d.purchases_30d IS NULL OR d.purchases_30d = 0 THEN 'NO_DEMAND'
        WHEN s.active_listings * 1.0 / d.purchases_30d > 100 THEN 'OVERSUPPLIED'
        WHEN s.active_listings * 1.0 / d.purchases_30d < 2 THEN 'UNDERSUPPLIED'
        ELSE 'BALANCED'
    END AS market_status
FROM supply s
LEFT JOIN demand d ON s.category = d.category
ORDER BY supply_demand_ratio DESC NULLS FIRST;
*/


-- ================================================================
-- PROBLEM 15: View → Watchlist → Purchase Sequence ⭐⭐⭐
-- ================================================================
-- Find users who viewed a listing, added it to cart, and then 
-- purchased it—in order—within 7 days. 
-- What % of cart additions result in a purchase?

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH view_cart_purchase AS (
    SELECT 
        ce.user_id,
        ce.listing_id,
        MIN(pv.view_time) AS first_view,
        MIN(ce.event_time) AS cart_add_time,
        MIN(t.transaction_date) AS purchase_time
    FROM cart_events ce
    JOIN page_views pv 
        ON ce.user_id = pv.user_id 
        AND ce.listing_id = pv.listing_id
        AND pv.view_time < ce.event_time
    LEFT JOIN transactions t 
        ON ce.user_id = t.buyer_id 
        AND ce.listing_id = t.listing_id
        AND t.status = 'completed'
        AND t.transaction_date > ce.event_time
        AND t.transaction_date <= ce.event_time + INTERVAL '7 days'
    WHERE ce.event_type = 'add'
    GROUP BY ce.user_id, ce.listing_id
)
SELECT 
    COUNT(*) AS total_cart_additions,
    COUNT(purchase_time) AS converted_to_purchase,
    ROUND(COUNT(purchase_time) * 100.0 / NULLIF(COUNT(*), 0), 2) AS cart_to_purchase_pct
FROM view_cart_purchase
WHERE first_view < cart_add_time;
*/


-- ================================================================
-- PROBLEM 16: Cannibalization Analysis ⭐⭐⭐⭐
-- ================================================================
-- When a seller lists a discounted variant (<80% of original price)
-- of an item, does it cannibalize sales of the full-price version?
-- Compare GMV 30 days before vs 30 days after the discount listing.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH discount_launches AS (
    SELECT 
        a.listing_id AS original_listing_id,
        b.listing_id AS discount_listing_id,
        a.seller_id,
        a.title,
        b.listing_date AS discount_launch_date,
        a.price AS original_price,
        b.price AS discount_price
    FROM listings a
    JOIN listings b 
        ON a.seller_id = b.seller_id
        AND a.category = b.category
        AND a.sub_category = b.sub_category
        AND a.listing_id != b.listing_id
        AND b.price < a.price * 0.8   -- discounted version
        AND b.listing_date > a.listing_date
),
before_after AS (
    SELECT 
        dl.original_listing_id,
        dl.discount_launch_date,
        SUM(CASE 
            WHEN t.transaction_date BETWEEN dl.discount_launch_date - INTERVAL '30 days' 
                                        AND dl.discount_launch_date
            THEN t.sale_price * t.quantity ELSE 0 
        END) AS gmv_before_30d,
        SUM(CASE 
            WHEN t.transaction_date BETWEEN dl.discount_launch_date 
                                        AND dl.discount_launch_date + INTERVAL '30 days'
            THEN t.sale_price * t.quantity ELSE 0 
        END) AS gmv_after_30d
    FROM discount_launches dl
    JOIN transactions t ON dl.original_listing_id = t.listing_id
    WHERE t.status = 'completed'
    GROUP BY dl.original_listing_id, dl.discount_launch_date
)
SELECT 
    original_listing_id,
    gmv_before_30d,
    gmv_after_30d,
    ROUND((gmv_after_30d - gmv_before_30d) * 100.0 / NULLIF(gmv_before_30d, 0), 2) 
        AS gmv_change_pct,
    CASE 
        WHEN gmv_after_30d < gmv_before_30d * 0.8 THEN 'LIKELY_CANNIBALIZED'
        ELSE 'NO_SIGNIFICANT_IMPACT'
    END AS impact
FROM before_after
ORDER BY gmv_change_pct ASC;
*/


-- ================================================================
-- PROBLEM 17: Click-Through Rate by Position ⭐⭐
-- ================================================================
-- For each search result position (1-10), calculate the average CTR.
-- CTR = #clicks at position / #times that position was shown.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH position_impressions AS (
    SELECT 
        sl.search_id,
        generate_series(1, LEAST(sl.num_results, 10)) AS position
    FROM search_logs sl
    WHERE sl.num_results > 0
),
position_clicks AS (
    SELECT 
        position, 
        COUNT(*) AS clicks
    FROM click_events
    WHERE position BETWEEN 1 AND 10
    GROUP BY position
),
position_shown AS (
    SELECT 
        position, 
        COUNT(*) AS impressions
    FROM position_impressions
    GROUP BY position
)
SELECT 
    ps.position,
    ps.impressions,
    COALESCE(pc.clicks, 0) AS clicks,
    ROUND(COALESCE(pc.clicks, 0) * 100.0 / NULLIF(ps.impressions, 0), 4) AS ctr_pct
FROM position_shown ps
LEFT JOIN position_clicks pc ON ps.position = pc.position
ORDER BY ps.position;
*/


-- ================================================================
-- PROBLEM 18: Seller Rating vs Sales Correlation ⭐⭐
-- ================================================================
-- For each seller, calculate avg rating and total GMV.
-- Group sellers into rating buckets (1-2, 2-3, 3-4, 4-5).
-- Show avg GMV per rating bucket.

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH seller_ratings AS (
    SELECT 
        seller_id,
        AVG(rating) AS avg_rating,
        COUNT(*) AS review_count
    FROM seller_reviews
    GROUP BY seller_id
    HAVING COUNT(*) >= 5  -- minimum reviews for reliability
),
seller_gmv AS (
    SELECT 
        seller_id,
        SUM(sale_price * quantity) AS total_gmv
    FROM transactions
    WHERE status = 'completed'
    GROUP BY seller_id
),
combined AS (
    SELECT 
        sr.seller_id,
        sr.avg_rating,
        COALESCE(sg.total_gmv, 0) AS total_gmv,
        CASE 
            WHEN sr.avg_rating < 2 THEN '1. Poor (1-2)'
            WHEN sr.avg_rating < 3 THEN '2. Below Avg (2-3)'
            WHEN sr.avg_rating < 4 THEN '3. Good (3-4)'
            ELSE '4. Excellent (4-5)'
        END AS rating_bucket
    FROM seller_ratings sr
    LEFT JOIN seller_gmv sg ON sr.seller_id = sg.seller_id
)
SELECT 
    rating_bucket,
    COUNT(*) AS num_sellers,
    ROUND(AVG(total_gmv), 2) AS avg_gmv_per_seller,
    ROUND(SUM(total_gmv), 2) AS total_bucket_gmv
FROM combined
GROUP BY rating_bucket
ORDER BY rating_bucket;
*/


-- ================================================================
-- PROBLEM 19: A/B Test Uplift Calculation (SQL) ⭐⭐⭐⭐
-- ================================================================
-- Given experiment results, calculate conversion lift and 
-- approximate p-value using a Z-test for proportions.

-- Assume table: experiments(user_id, variant, converted, revenue)

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH stats AS (
    SELECT 
        variant,
        COUNT(*) AS n,
        SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) AS conversions,
        SUM(CASE WHEN converted = 1 THEN 1 ELSE 0 END) * 1.0 / COUNT(*) AS conv_rate
    FROM experiments
    GROUP BY variant
),
pooled AS (
    SELECT 
        c.conv_rate AS control_rate,
        t.conv_rate AS treatment_rate,
        c.n AS n_control,
        t.n AS n_treatment,
        (c.conversions + t.conversions) * 1.0 / (c.n + t.n) AS pooled_rate,
        t.conv_rate - c.conv_rate AS absolute_lift,
        (t.conv_rate - c.conv_rate) / NULLIF(c.conv_rate, 0) AS relative_lift
    FROM stats c, stats t
    WHERE c.variant = 'control' AND t.variant = 'treatment'
)
SELECT 
    control_rate,
    treatment_rate,
    ROUND(absolute_lift * 100, 4) AS absolute_lift_pct,
    ROUND(relative_lift * 100, 2) AS relative_lift_pct,
    ROUND((treatment_rate - control_rate) / 
        SQRT(pooled_rate * (1 - pooled_rate) * (1.0/n_control + 1.0/n_treatment)), 4
    ) AS z_score
    -- z_score > 1.96 → significant at α=0.05
FROM pooled;
*/


-- ================================================================
-- PROBLEM 20: Device-Category Purchase Matrix ⭐⭐
-- ================================================================
-- Create a pivot table showing purchase count by device type 
-- (rows) and top 5 categories (columns).

-- YOUR SOLUTION:


-- ANSWER:
/*
WITH top_categories AS (
    SELECT l.category
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    WHERE t.status = 'completed'
    GROUP BY l.category
    ORDER BY COUNT(*) DESC
    LIMIT 5
),
purchase_with_device AS (
    SELECT 
        pv.device_type,
        l.category,
        COUNT(DISTINCT t.transaction_id) AS purchase_count
    FROM transactions t
    JOIN listings l ON t.listing_id = l.listing_id
    JOIN page_views pv 
        ON t.buyer_id = pv.user_id 
        AND t.listing_id = pv.listing_id
    WHERE t.status = 'completed'
      AND l.category IN (SELECT category FROM top_categories)
    GROUP BY pv.device_type, l.category
)
SELECT 
    device_type,
    MAX(CASE WHEN category = (SELECT category FROM top_categories LIMIT 1 OFFSET 0) THEN purchase_count END) AS cat_1,
    MAX(CASE WHEN category = (SELECT category FROM top_categories LIMIT 1 OFFSET 1) THEN purchase_count END) AS cat_2,
    MAX(CASE WHEN category = (SELECT category FROM top_categories LIMIT 1 OFFSET 2) THEN purchase_count END) AS cat_3,
    MAX(CASE WHEN category = (SELECT category FROM top_categories LIMIT 1 OFFSET 3) THEN purchase_count END) AS cat_4,
    MAX(CASE WHEN category = (SELECT category FROM top_categories LIMIT 1 OFFSET 4) THEN purchase_count END) AS cat_5
FROM purchase_with_device
GROUP BY device_type;
*/

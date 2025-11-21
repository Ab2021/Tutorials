# Day 41 Interview Prep: Payment Systems

## Q1: Why use UUID for IDs instead of Auto-Increment?
**Answer:**
*   **Security:** Auto-increment reveals business volume (`order_id=100` -> `101`).
*   **Distributed:** Can generate UUIDs without coordination (Snowflake).
*   **Merging:** Easier to merge databases later.

## Q2: How to handle floating point errors?
**Answer:**
*   **Never** use `float` or `double` for money. (`0.1 + 0.2 = 0.300000004`).
*   **Use:**
    *   `Decimal` type (in SQL/Python).
    *   **Integers (Cents):** Store $10.50 as `1050` cents.

## Q3: What if the PSP is down?
**Answer:**
*   **Circuit Breaker:** Stop calling PSP.
*   **Queue:** Queue the payment requests (if user is okay with async processing).
*   **Fallback:** Route to a backup PSP (e.g., Stripe down -> Switch to Braintree). This requires "Smart Routing" logic.

## Q4: Explain the difference between Authorization and Capture.
**Answer:**
*   **Auth:** "Hold" the money. Verifies funds exist. (e.g., Hotel check-in).
*   **Capture:** Actually move the money. (e.g., Hotel check-out).
*   **Void:** Cancel the Auth.

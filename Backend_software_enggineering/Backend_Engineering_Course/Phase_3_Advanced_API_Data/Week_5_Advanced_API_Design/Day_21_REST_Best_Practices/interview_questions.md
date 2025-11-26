# Day 21: Interview Questions & Answers

## Conceptual Questions

### Q1: Why is Offset Pagination slow for large datasets?
**Answer:**
*   **Mechanism**: `OFFSET 1000000 LIMIT 10`.
*   **DB Action**: The database must fetch 1,000,010 rows, throw away the first 1,000,000, and return the last 10.
*   **Cost**: It scans the index/table for all previous rows. `O(N)` where N is the offset.
*   **Fix**: Cursor pagination (`WHERE id > 1000000 LIMIT 10`) uses the index to jump directly. `O(1)` (or `O(Limit)`).

### Q2: What is the difference between `PUT` and `PATCH`?
**Answer:**
*   **PUT**: **Replace**. You send the *entire* resource. If fields are missing, they should be nulled out (technically).
*   **PATCH**: **Partial Update**. You send only the fields that changed.
*   *Note*: In practice, many APIs implement "Partial PUT", but that violates REST semantics.

### Q3: How do you implement Idempotency for a Payment API?
**Answer:**
1.  Client sends `Idempotency-Key` header (UUID).
2.  Server checks a distributed lock/cache (Redis) for that key.
3.  **If Found**: Return the stored response (200 OK or Error) immediately. Do not process again.
4.  **If Not Found**:
    *   Lock the key.
    *   Process payment.
    *   Store response in Redis with TTL (e.g., 24 hours).
    *   Return response.

---

## Scenario-Based Questions

### Q4: A client complains that `GET /users` is timing out. It returns 50,000 rows. How do you fix it?
**Answer:**
*   **Immediate**: Force pagination. Set a default `limit=50` if not provided. Hard cap `limit=100`.
*   **Communication**: Return `413 Payload Too Large` or just the first page with a `Link` header for the next page.
*   **Long-term**: If they *need* all data, provide an Export API (Async job that generates a CSV and emails a link).

### Q5: You need to design a search endpoint that supports complex logic like `(age > 20 OR role = 'admin') AND status = 'active'`. How do you map this to URL params?
**Answer:**
*   **Simple Params**: `?age_gt=20` works for AND, but fails for OR/Grouping.
*   **Solution**:
    1.  **RSQL / FIQL**: A standard syntax. `?q=age>20,role=='admin';status=='active'`.
    2.  **JSON Filter**: `?filter={"$or": [{"age": {"$gt": 20}}, {"role": "admin"}], "status": "active"}`. (Harder to read/encode).
    3.  **POST Search**: Move to `POST /users/search` and send the complex filter in the body. (Violates "GET for reads", but pragmatic).

---

## Behavioral / Role-Specific Questions

### Q6: A frontend developer asks you to include the "User's Last 5 Orders" in the `GET /users/{id}` response to save an API call. Do you agree?
**Answer:**
*   **Trade-off**:
    *   *Yes*: Better performance for that specific UI screen (BFF pattern).
    *   *No*: Bloats the User resource. What if the Orders service is down? Now User service fails too.
*   **Decision**:
    *   If it's a generic Public API: **No**. Keep resources pure. Use `embed` or `expand` param if supported.
    *   If it's a BFF (Internal): **Yes**. That's the point of a BFF.

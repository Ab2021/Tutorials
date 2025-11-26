# Day 21: REST API Best Practices

## 1. Beyond CRUD

Anyone can build `GET /users`. But how do you handle 1 million users? Or complex search? Or errors?

### 1.1 Pagination
Never return a full list.
1.  **Offset Pagination**: `GET /items?limit=10&offset=20`
    *   *Pros*: Simple, easy to jump to page 10.
    *   *Cons*: Slow for large offsets (DB must scan and discard rows). Unstable if items are added/deleted while paging.
2.  **Cursor Pagination**: `GET /items?limit=10&cursor=eyJpZCI6MTB9`
    *   *Pros*: Infinite scale (`WHERE id > last_seen_id`). Stable.
    *   *Cons*: Cannot jump to page 10. Harder to implement.
    *   *Verdict*: Use Cursor for feeds/infinite scroll. Use Offset for admin tables.

### 1.2 Filtering & Sorting
*   **Filtering**: `GET /users?role=admin&age_gt=25`
    *   Use LHS brackets for operators: `price[gte]=100`.
*   **Sorting**: `GET /users?sort=-created_at,name`
    *   `-` for descending, `+` (or nothing) for ascending.

### 1.3 Field Selection (Partial Response)
*   `GET /users?fields=id,name`
*   Saves bandwidth for mobile clients.

---

## 2. Error Handling (RFC 7807)

Don't just return `400 Bad Request`. Return structured details.
**Problem Details for HTTP APIs**:
```json
{
  "type": "https://example.com/probs/out-of-credit",
  "title": "You do not have enough credit.",
  "status": 403,
  "detail": "Your current balance is 30, but that costs 50.",
  "instance": "/account/12345/msgs/abc"
}
```
*   **Type**: A URI reference to documentation.
*   **Title**: Short summary.
*   **Detail**: Human-readable explanation.

---

## 3. Idempotency

*   **Definition**: Making the same request multiple times has the same effect as making it once.
*   **Safe Methods**: `GET`, `PUT`, `DELETE` are idempotent by definition.
*   **Unsafe**: `POST` is NOT idempotent. (Charging a card twice).
*   **Idempotency Keys**:
    1.  Client generates a UUID `Idempotency-Key: <uuid>`.
    2.  Server checks Redis. If key exists, return cached response.
    3.  If not, process and save response.

---

## 4. HATEOAS (Hypermedia)

**Hypermedia As The Engine Of Application State**.
The API tells the client what it can do next.
```json
{
  "id": 1,
  "status": "pending",
  "links": [
    { "rel": "self", "href": "/orders/1", "method": "GET" },
    { "rel": "cancel", "href": "/orders/1/cancel", "method": "POST" },
    { "rel": "pay", "href": "/orders/1/pay", "method": "POST" }
  ]
}
```
*   *Benefit*: The client doesn't need to hardcode logic ("If status is pending, show cancel button"). It just renders the links.

---

## 5. Summary

Today we polished our API.
*   **Pagination**: Cursor for scale.
*   **Errors**: RFC 7807.
*   **Idempotency**: Essential for payments.

**Tomorrow (Day 22)**: We will learn how to change our API without breaking clients using **Versioning Strategies**.

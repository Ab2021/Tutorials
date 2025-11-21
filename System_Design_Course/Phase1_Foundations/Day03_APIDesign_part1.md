# Day 3 Deep Dive: API Best Practices

## 1. Versioning
*   **URI Versioning:** `/v1/users`. (Most common, easy to cache).
*   **Header Versioning:** `Accept: application/vnd.myapi.v1+json`. (Cleaner URLs, harder to test).
*   **Query Param:** `/users?version=1`.

## 2. Pagination
*   **Offset-based:** `LIMIT 10 OFFSET 100`.
    *   **Pros:** Easy to implement.
    *   **Cons:** Slow on large datasets (`OFFSET` scans rows). Unstable if items are added/deleted.
*   **Cursor-based:** `LIMIT 10 WHERE id > cursor_id`.
    *   **Pros:** Efficient (Index seek). Stable.
    *   **Cons:** Harder to implement "Jump to Page 10".

## 3. Rate Limiting Headers
Always inform the client:
*   `X-RateLimit-Limit`: 1000
*   `X-RateLimit-Remaining`: 999
*   `X-RateLimit-Reset`: 1600000000

## 4. Error Handling
Use standard HTTP Codes:
*   **200:** OK.
*   **201:** Created.
*   **400:** Bad Request (Client error).
*   **401:** Unauthorized (No token).
*   **403:** Forbidden (Has token, but no permission).
*   **404:** Not Found.
*   **429:** Too Many Requests.
*   **500:** Internal Server Error.

## 5. Security
*   **HTTPS:** Always.
*   **Authentication:** JWT (Stateless) vs Sessions (Stateful).
*   **CORS:** Configure `Access-Control-Allow-Origin`.

# Day 3: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the difference between `PUT` and `PATCH`?
**Answer:**
*   **PUT**: Replaces the **entire** resource. If you send `{ "email": "new@a.com" }` to a PUT endpoint, fields not included (like `username`) should theoretically be nulled out or reset to default (though many APIs implement this loosely). It is **Idempotent**.
*   **PATCH**: Applies a **partial** update. Only the fields sent in the body are modified. It is generally **Not Idempotent** (though it can be designed to be).

### Q2: Explain "Idempotency" with a real-world example.
**Answer:**
Idempotency means that making the same request multiple times has the same effect as making it once.
*   **Example**: A payment API.
    *   *Non-Idempotent*: `POST /pay { amount: 100 }`. If the network times out but the server processed it, retrying will charge the user another $100.
    *   *Idempotent*: `POST /pay { amount: 100, idempotency_key: "uuid-123" }`. If the client retries with the same key, the server sees it already processed that key and returns the previous success response without charging again.

### Q3: What is the difference between HTTP/1.1 and HTTP/2?
**Answer:**
*   **Text vs Binary**: HTTP/1.1 is text-based; HTTP/2 is binary (more efficient parsing).
*   **Multiplexing**: HTTP/1.1 has Head-of-Line blocking (one request per connection at a time). HTTP/2 allows multiple streams (requests) over a single TCP connection simultaneously.
*   **Header Compression**: HTTP/2 uses HPACK to compress headers, reducing overhead.

---

## Scenario-Based Questions

### Q4: You see a spike in `502 Bad Gateway` errors in your load balancer logs. What does this mean and how do you debug it?
**Answer:**
*   **Meaning**: The Load Balancer (Gateway) tried to talk to the upstream backend service, but the backend sent an invalid response or closed the connection abruptly.
*   **Debugging Steps**:
    1.  **Check Backend Health**: Is the backend service crashing? (Check `docker ps` or K8s pod restarts).
    2.  **Check Logs**: Look at the backend logs. Is it panicking? Is it timing out connecting to the DB?
    3.  **Timeouts**: Did the request take too long? If the LB timeout is 30s and the backend took 31s, the LB returns 502.
    4.  **Resources**: Is the backend OOM (Out of Memory)?

### Q5: A mobile developer asks you to change a `404 Not Found` to a `200 OK` with `body: null` because "handling errors is annoying on the client". Do you agree?
**Answer:**
**No.**
*   **Semantics**: HTTP status codes are the standard contract. `404` explicitly means "resource missing". `200` means "success".
*   **Observability**: Monitoring tools track error rates based on 4xx/5xx codes. If we return 200 for errors, our dashboards will show 100% success rate while users are seeing blank screens.
*   **Caching**: Proxies and CDNs cache 200 responses differently than 404s.
*   *Compromise*: If they need a specific error format, I will return `404` with a standard JSON body `{ "error": "UserNotFound", "message": "..." }`.

---

## Behavioral / Role-Specific Questions

### Q6: Explain the "Stateless" constraint of REST to a junior developer. Why is it important for scaling?
**Answer:**
"Stateless" means the server does not remember the client's previous requests. It doesn't store 'session data' in its own memory (RAM).
*   **Why it matters**: Imagine we have 3 servers (A, B, C).
    *   If Server A stores "User is logged in" in its RAM, and the next request goes to Server B, Server B won't know the user.
    *   To fix this, we'd need "Sticky Sessions" (routing user to A always), which creates hotspots and makes autoscaling hard.
*   **The Stateless Way**: The client sends the session info (e.g., JWT Token) with *every* request. Any server (A, B, or C) can verify the token and process the request. This allows us to add/remove servers instantly.

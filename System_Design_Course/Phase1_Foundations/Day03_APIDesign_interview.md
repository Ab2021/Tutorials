# Day 3 Interview Prep: API Design

## Q1: REST vs GraphQL?
**Answer:**
*   **REST:** Standard, cacheable, simple. Good for public APIs. Prone to over-fetching (getting too much data) or under-fetching (needing multiple calls).
*   **GraphQL:** Flexible, single endpoint. Client decides what it wants. Solves over/under-fetching. Harder to cache (everything is POST). Good for complex frontends (Facebook).

## Q2: How to handle long-running processes in API?
**Answer:**
*   **Async Pattern:**
    1.  Client sends POST `/export`.
    2.  Server returns `202 Accepted` with a `Location: /jobs/123`.
    3.  Client polls `/jobs/123` until status is `COMPLETED`.
    4.  Or use Webhooks to notify client.

## Q3: What is Idempotency and why is it important?
**Answer:**
*   **Definition:** $f(f(x)) = f(x)$.
*   **Importance:** In distributed systems, networks fail. If a client retries a payment request, we must not charge them twice.
*   **Mechanism:** Use a unique `idempotency_key`. Server tracks processed keys.

## Q4: Design an API for a News Feed.
**Answer:**
*   `GET /feed?limit=10&cursor=abc`: Get posts.
*   `POST /posts`: Create post.
*   `POST /posts/{id}/like`: Like a post.
*   `POST /posts/{id}/comment`: Comment.
*   **Key:** Use Cursor pagination for infinite scroll.

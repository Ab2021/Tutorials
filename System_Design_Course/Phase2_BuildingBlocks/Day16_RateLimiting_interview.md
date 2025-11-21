# Day 16 Interview Prep: Rate Limiting

## Q1: Token Bucket vs Leaky Bucket?
**Answer:**
*   **Token Bucket:** Allows bursts. (Good for user actions like clicking buttons).
*   **Leaky Bucket:** Enforces constant rate. (Good for packet shaping / network traffic).

## Q2: How to handle distributed rate limiting without Redis latency?
**Answer:**
*   **Batching:** Client requests 10 tokens at once.
*   **Local Cache:** Allow 100 req/s locally, sync with Redis asynchronously (Loose consistency).

## Q3: What HTTP Headers should you return?
**Answer:**
*   `429 Too Many Requests`
*   `X-RateLimit-Limit`
*   `X-RateLimit-Remaining`
*   `X-RateLimit-Reset` (Timestamp when limit resets).

## Q4: Design a Rate Limiter for a global system.
**Answer:**
*   **Edge:** Rate limit at the CDN/Edge (Cloudflare) to stop DDoS early.
*   **Service:** Rate limit per user/IP at the API Gateway.
*   **DB:** Rate limit internal services to protect the Database (Circuit Breaker).

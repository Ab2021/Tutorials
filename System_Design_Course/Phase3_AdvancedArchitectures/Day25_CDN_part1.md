# Day 25 Deep Dive: Edge Computing

## 1. Anycast DNS
*   **Unicast:** One IP = One Server.
*   **Anycast:** One IP = Many Servers globally.
*   **Mechanism:** BGP (Border Gateway Protocol). Routers send traffic to the "topologically nearest" server announcing that IP.
*   **Benefit:** Instant failover. If London data center dies, BGP routes traffic to Paris automatically.

## 2. Edge Computing (Cloudflare Workers / AWS Lambda@Edge)
*   **Concept:** Run code at the CDN Edge (not just static files).
*   **Use Cases:**
    *   **Auth:** Validate JWT at the edge. Reject 401 before hitting Origin.
    *   **A/B Testing:** Serve Variant A or B based on cookie.
    *   **Image Resizing:** Resize on the fly based on User-Agent.
*   **Benefit:** Extremely low latency (< 50ms).

## 3. Cache Invalidation
*   **Problem:** You updated `style.css` but users see old version.
*   **Strategies:**
    *   **TTL (Time To Live):** Set `Cache-Control: max-age=3600`. (Wait 1 hour).
    *   **Purge:** API call to clear cache. (Expensive, takes seconds/minutes).
    *   **Versioning (Best):** `style.v2.css`. Immutable files. Never invalidate, just change the link.

## 4. Case Study: Netflix Open Connect
*   Netflix built its own CDN (ISP-embedded).
*   They ship physical boxes (Appliances) to ISPs (Comcast, Verizon).
*   **Benefit:** Traffic doesn't leave the ISP network. Free peering.

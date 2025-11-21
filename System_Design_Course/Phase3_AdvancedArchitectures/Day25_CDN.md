# Day 25: Content Delivery Networks (CDN)

## 1. What is a CDN?
A distributed network of servers (PoPs - Points of Presence) that deliver content to users based on their geographic location.
*   **Goal:** Reduce latency (speed of light) and offload Origin server.

## 2. How it works
1.  User requests `image.png`.
2.  DNS routes user to nearest Edge Server (e.g., London).
3.  **Cache Hit:** Edge returns image. (10ms).
4.  **Cache Miss:** Edge fetches from Origin (US), caches it, returns it. (200ms).

## 3. Types of Content
*   **Static:** Images, CSS, JS, Video. (Easy to cache).
*   **Dynamic:** Personalized API responses. (Hard to cache).
    *   **Dynamic Acceleration:** CDN optimizes the route (TCP optimization) back to Origin, but doesn't cache content.

## 4. Push vs Pull
*   **Pull (Standard):** CDN fetches from Origin on miss.
*   **Push:** You upload files directly to CDN. (Good for large files/Software updates).

## 5. Providers
*   Cloudflare, Akamai, AWS CloudFront, Fastly.

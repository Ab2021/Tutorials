# Day 2 Interview Prep: Networking

## Q1: What happens when you type google.com in your browser?
**Answer:**
1.  **DNS Lookup:** Browser checks cache -> OS cache -> ISP DNS -> Root DNS -> TLD DNS -> Authoritative DNS. Returns IP.
2.  **TCP Handshake:** SYN, SYN-ACK, ACK. Connection established.
3.  **TLS Handshake:** Exchange certificates, keys. Encrypted tunnel.
4.  **HTTP Request:** GET / index.html.
5.  **Server Processing:** Load Balancer -> Web Server -> DB.
6.  **HTTP Response:** 200 OK + HTML.
7.  **Rendering:** Browser parses HTML, fetches CSS/JS, renders DOM.

## Q2: TCP vs UDP?
**Answer:**
*   **TCP:** Reliable, connection-oriented. Use for critical data (Banking, Web).
*   **UDP:** Unreliable, connectionless, fast. Use for real-time (VoIP, FPS Games).

## Q3: Why is HTTP/2 faster than HTTP/1.1?
**Answer:**
*   **Multiplexing:** Multiple requests over a single TCP connection.
*   **Header Compression:** HPACK reduces overhead.
*   **Server Push:** Server sends resources (CSS/JS) before client asks.

## Q4: What is a Sticky Session?
**Answer:**
*   A Load Balancer feature where requests from the same client always go to the same server.
*   **Pros:** Easy to use local memory cache.
*   **Cons:** Hard to scale. If server dies, session is lost. Uneven load distribution.
*   **Better:** Stateless servers + Redis.

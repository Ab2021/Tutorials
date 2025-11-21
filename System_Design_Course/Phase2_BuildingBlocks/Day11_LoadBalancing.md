# Day 11: Load Balancing

## 1. What is Load Balancing?
The process of distributing network traffic across multiple servers to ensure no single server is overwhelmed.
*   **Goal:** Maximize throughput, minimize response time, avoid overload.

## 2. L4 vs L7 Load Balancing
### Layer 4 (Transport Layer)
*   **Data:** IP Address + Port (TCP/UDP).
*   **Mechanism:** NAT (Network Address Translation). Modifies dest IP and forwards packet.
*   **Pros:** Extremely fast, secure (doesn't decrypt SSL).
*   **Cons:** "Dumb". Can't see URL, Headers, or Cookies.
*   **Example:** LVS (Linux Virtual Server).

### Layer 7 (Application Layer)
*   **Data:** HTTP Headers, URL, Cookies.
*   **Mechanism:** Terminates TCP connection, reads request, decides destination, opens new connection to backend.
*   **Pros:** Smart routing (e.g., `/api` to Service A, `/static` to Service B). Can do SSL Termination.
*   **Cons:** Slower (CPU intensive).
*   **Example:** Nginx, HAProxy, AWS ALB.

## 3. Algorithms
*   **Round Robin:** Sequential (A, B, C, A...). Good for equal servers.
*   **Weighted Round Robin:** A (Weight 2), B (Weight 1). -> A, A, B. Good for heterogeneous servers.
*   **Least Connections:** Send to server with fewest active connections. Good for long-lived connections (Chat).
*   **Least Response Time:** Send to server with fastest response.
*   **IP Hash:** `hash(ClientIP) % N`. Ensures client always goes to same server (Sticky Session).

## 4. Health Checks
*   **Active:** LB pings server (`/health`). If 200 OK, it's alive. If 500 or Timeout, remove from pool.
*   **Passive:** LB observes real traffic. If 502 errors spike, remove server.

# Day 11 Interview Prep: Load Balancing

## Q1: L4 vs L7 Load Balancing?
**Answer:**
*   **L4:** Transport layer. Routes based on IP/Port. Fast, dumb. No SSL termination.
*   **L7:** Application layer. Routes based on URL/Headers. Slower, smart. SSL termination.

## Q2: How does a Load Balancer handle SSL?
**Answer:**
*   **SSL Termination:** LB decrypts the traffic. Traffic between LB and Backend is HTTP (Unencrypted).
    *   **Pros:** Offloads CPU work from backend. Centralized cert management.
    *   **Cons:** Security risk if internal network is compromised.
*   **SSL Passthrough:** LB just forwards encrypted packets (L4). Backend decrypts.
    *   **Pros:** End-to-End encryption.
    *   **Cons:** No smart routing (LB can't see headers).

## Q3: What happens if the Load Balancer dies?
**Answer:**
*   **Active-Passive HA:** Run two LBs.
    *   **Active:** Handles traffic. Sends heartbeats to Passive.
    *   **Passive:** Listens. If heartbeat stops, takes over the Virtual IP (VIP) using ARP/VRRP.
*   **Keepalived:** A tool that implements VRRP for Linux LBs.

## Q4: Design a Load Balancer for a Chat App.
**Answer:**
*   Chat requires long-lived connections (WebSockets).
*   **Algorithm:** Least Connections (Round Robin is bad because one connection might last hours).
*   **Stickiness:** IP Hash (to keep user on same server) OR store session in Redis (Stateless).

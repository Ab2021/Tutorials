# Day 13: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the main risk of using an API Gateway?
**Answer:**
*   **Single Point of Failure (SPOF)**: If the Gateway goes down, the entire system is inaccessible, even if the microservices are healthy.
*   **Bottleneck**: All traffic passes through it. It must be highly performant and scalable.
*   **Complexity**: It adds another hop (latency) and another component to manage.

### Q2: Explain the "Token Bucket" algorithm for rate limiting.
**Answer:**
*   **Concept**: You have a bucket that holds `N` tokens.
*   **Refill**: Tokens are added at a constant rate `R` (e.g., 10 tokens/sec).
*   **Consume**: Each request removes 1 token.
*   **Logic**: If the bucket is empty, the request is rejected (429 Too Many Requests). If full, new tokens spill over (discarded).
*   **Benefit**: Allows for short **bursts** of traffic (up to bucket size) while enforcing a long-term average rate.

### Q3: Why use a BFF (Backend for Frontend) instead of a generic API?
**Answer:**
*   **Optimization**: Mobile devices have limited bandwidth/battery. A BFF can aggregate 3 API calls into 1 and strip unnecessary data.
*   **Coupling**: If the Mobile UI changes, we only update the Mobile BFF. The core microservices remain untouched.
*   **Team Autonomy**: The Mobile team can own the Mobile BFF (Node.js) and iterate fast without waiting for the Backend team (Java/Go).

---

## Scenario-Based Questions

### Q4: You have a global user base. How do you deploy your API Gateway?
**Answer:**
*   **Edge Deployment**: Deploy the Gateway close to the user (e.g., AWS CloudFront / Cloudflare Workers or multi-region K8s clusters).
*   **Global Routing**: Use DNS (Latency-based routing) to direct users to the nearest Gateway.
*   **Caching**: The Gateway should cache static responses at the edge to reduce latency.

### Q5: Your Gateway handles SSL termination. Is the traffic between the Gateway and Microservices unencrypted (HTTP)?
**Answer:**
*   **Traditionally**: Yes, for performance (SSL offloading). Inside the VPC, we trust the network.
*   **Zero Trust (Modern)**: No. We use **mTLS** (Mutual TLS) via a Service Mesh (Istio/Linkerd). The Gateway encrypts traffic to the service, ensuring that even if an attacker gets into the VPC, they can't sniff traffic.

---

## Behavioral / Role-Specific Questions

### Q6: A developer wants to put business logic (e.g., "If user is VIP, add discount") in the API Gateway. Do you approve?
**Answer:**
**No.**
*   **Reasoning**: The Gateway should be "Dumb Pipes". Its job is Routing, Auth, and Throttling.
*   **Risk**: Putting logic in the Gateway makes it a "God Object". It becomes hard to test, hard to deploy, and couples business rules to infrastructure.
*   **Place it**: Put the logic in the `Order Service` or a `Pricing Service`. The Gateway just passes the user's role.

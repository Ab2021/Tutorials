# Day 18: Service Discovery

## 1. The Problem
*   **Monolith:** `localhost:8080`. Easy.
*   **Microservices:** Service A needs to call Service B. Service B has 100 nodes. IPs change dynamically (Auto-scaling).
*   **Hardcoding IPs:** Impossible.

## 2. Approaches
### Client-Side Discovery
*   **Client** queries Registry to get list of IPs.
*   **Client** picks one (Load Balancing).
*   **Pros:** Less hops.
*   **Cons:** Client needs logic for every language.
*   **Example:** Netflix Eureka + Ribbon.

### Server-Side Discovery
*   **Client** calls Load Balancer (DNS name).
*   **LB** queries Registry and forwards traffic.
*   **Pros:** Simple client.
*   **Cons:** LB is bottleneck.
*   **Example:** AWS ELB, Kubernetes Service.

## 3. The Registry
*   **Database of Services:** `Service B -> [IP1, IP2, IP3]`.
*   **Health Checks:** Registry must remove dead nodes.
*   **Tools:** Zookeeper, Etcd, Consul, Eureka.

## 4. DNS as Discovery?
*   **Pros:** Simple.
*   **Cons:** DNS caching (TTL). If IP changes, client might hold old IP for minutes.

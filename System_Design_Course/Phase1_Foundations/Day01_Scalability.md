# Day 1: Scalability Basics

## 1. What is Scalability?
Scalability is the ability of a system to handle increased load (users, data, traffic) without performance degradation.

## 2. Vertical vs Horizontal Scaling
### Vertical Scaling (Scale Up)
*   **Definition:** Adding more power (CPU, RAM, SSD) to an existing server.
*   **Pros:** Simple (no code changes), fast to implement.
*   **Cons:** Hardware limits (ceiling), Single Point of Failure (SPOF), expensive at high end.
*   **Use Case:** Early stage startups, small databases.

### Horizontal Scaling (Scale Out)
*   **Definition:** Adding more servers (nodes) to the pool.
*   **Pros:** Infinite scaling (theoretically), redundancy/fault tolerance, cost-effective (commodity hardware).
*   **Cons:** Complex (requires load balancing, distributed data consistency, network overhead).
*   **Use Case:** Google, Facebook, Amazon, any large-scale system.

## 3. Key Metrics
*   **Throughput:** Number of operations per second (RPS/QPS).
*   **Latency:** Time taken to process a request (p50, p95, p99).
*   **Availability:** Uptime (99.9%, 99.99%).
*   **Consistency:** Data uniformity across nodes.

## 4. The Scalability Pyramid
1.  **Load Balancer:** Distributes traffic.
2.  **Stateless Services:** Web/App servers should not store state locally.
3.  **Caching:** Reduce DB load (Redis/Memcached).
4.  **Database Replication:** Read replicas for read-heavy workloads.
5.  **Database Sharding:** Partitioning data for write-heavy workloads.

## 5. Code Example: Simple Load Balancer (Python)
```python
import random

class LoadBalancer:
    def __init__(self, servers):
        self.servers = servers
    
    def get_server_round_robin(self, request_id):
        # Simple Round Robin
        return self.servers[request_id % len(self.servers)]
    
    def get_server_random(self):
        return random.choice(self.servers)

servers = ["Server1", "Server2", "Server3"]
lb = LoadBalancer(servers)

for i in range(5):
    print(f"Request {i} -> {lb.get_server_round_robin(i)}")
```

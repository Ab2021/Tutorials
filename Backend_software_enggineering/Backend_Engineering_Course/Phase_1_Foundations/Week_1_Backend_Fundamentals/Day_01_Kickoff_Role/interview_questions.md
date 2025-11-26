# Day 1: Interview Questions & Answers

## Conceptual Questions

### Q1: What is the CAP theorem, and why is it relevant to modern backend systems?
**Answer:**
The CAP theorem states that a distributed data store can only provide two of the following three guarantees simultaneously:
1.  **Consistency (C)**: Every read receives the most recent write or an error.
2.  **Availability (A)**: Every request receives a (non-error) response, without the guarantee that it contains the most recent write.
3.  **Partition Tolerance (P)**: The system continues to operate despite an arbitrary number of messages being dropped or delayed by the network between nodes.

**Relevance:** In modern distributed systems (cloud environments), network partitions are inevitable (P is mandatory). Therefore, architects must choose between Consistency (CP) and Availability (AP).
- **CP**: Choose when data accuracy is critical (e.g., banking transactions). If the network splits, the system stops accepting writes to prevent divergence.
- **AP**: Choose when uptime is critical (e.g., social media feed). If the network splits, the system keeps accepting writes, but data might be temporarily stale (eventual consistency).

### Q2: Explain the difference between Vertical Scaling and Horizontal Scaling.
**Answer:**
- **Vertical Scaling (Scaling Up)**: Adding more power (CPU, RAM, Disk) to an existing machine.
    - *Pros*: Simple, no code changes usually required.
    - *Cons*: Hardware limits (there's a max RAM size), single point of failure, expensive at the high end.
- **Horizontal Scaling (Scaling Out)**: Adding more machines (nodes) to the system pool.
    - *Pros*: Theoretically infinite scale, redundancy/high availability.
    - *Cons*: Increased complexity (data partitioning, consistency issues, network latency).

### Q3: What is "Eventual Consistency"?
**Answer:**
Eventual consistency is a consistency model used in distributed computing to achieve high availability. It guarantees that if no new updates are made to a given data item, eventually all accesses to that item will return the last updated value. It allows for temporary inconsistencies between replicas to prioritize low latency and availability.

---

## Scenario-Based Questions

### Q4: You are designing a backend for a global e-commerce platform. During a "Black Friday" sale, traffic spikes 100x. Your database is the bottleneck. How do you approach this?
**Answer:**
This is a classic scaling problem.
1.  **Immediate Mitigation**:
    - **Read Replicas**: If the load is read-heavy (viewing products), spin up read replicas and offload read traffic from the primary DB.
    - **Caching**: Implement aggressive caching (Redis/CDN) for static content and product details.
2.  **Architecture Changes**:
    - **Queueing**: Introduce a message queue (Kafka/SQS) for write-heavy operations like "Place Order". Process orders asynchronously to flatten the spike.
    - **Sharding**: Partition the database by user ID or region to distribute write load across multiple nodes.
3.  **Degradation**: Implement "graceful degradation". If the recommendation engine is slow, disable it temporarily to keep the checkout flow alive.

### Q5: A junior engineer suggests using a single large machine for the database to avoid the complexity of distributed systems. How do you respond?
**Answer:**
I would acknowledge that their intuition about simplicity is correct—distributed systems *are* complex. However, I would highlight the risks:
1.  **Single Point of Failure (SPOF)**: If that one machine dies (hardware failure, OS crash), the entire business is down.
2.  **Scalability Ceiling**: Eventually, we will hit the physical limits of the largest available machine.
3.  **Maintenance**: We can't patch the OS or upgrade the DB without downtime.
**Conclusion**: We should start simple (maybe one primary + one standby for failover), but design the application to be stateless so we can scale the app tier horizontally, and plan for DB read replicas as we grow.

---

## Behavioral / Role-Specific Questions

### Q6: How do you keep up with the rapidly changing backend landscape (e.g., AI integration, new tools)?
**Answer:**
*Key points to hit:*
- **Curiosity**: I regularly read engineering blogs (Uber, Netflix, Cloudflare) to see how big players solve problems.
- **Hands-on**: I build side projects to try new tech (e.g., recently built a RAG pipeline to understand vector DBs).
- **Discernment**: I don't jump on every hype train. I evaluate tools based on trade-offs (complexity vs. benefit) and stability.

### Q7: Describe a time you caused a production outage. How did you handle it?
**Answer:**
*Use the STAR method (Situation, Task, Action, Result).*
- **Situation**: I deployed a database migration that locked a critical table for too long during peak hours.
- **Task**: The site became unresponsive. I needed to restore service immediately.
- **Action**: I immediately rolled back the deployment (or killed the query). Once service was restored, I investigated why the migration locked the table (it was a non-concurrent index creation on a large table).
- **Result**: I rewrote the migration to be non-blocking (using `CONCURRENTLY` in Postgres). I also instituted a new policy where migrations are reviewed by a DBA and tested on a staging DB with production-scale data volume.

# Day 1 Interview Prep: Scalability

## Q1: Vertical vs Horizontal Scaling?
**Answer:**
*   **Vertical (Scale Up):** Add CPU/RAM to one machine. Easy, but has a hard limit (ceiling) and is a Single Point of Failure. Good for small apps or DBs where consistency is key and sharding is hard.
*   **Horizontal (Scale Out):** Add more machines. Infinite scaling, fault tolerant, cheaper hardware. Harder to manage (distributed state, consistency). The standard for large systems.

## Q2: What are the numbers you should memorize for system design?
**Answer:**
*   L1 Cache: 0.5ns
*   Mutex Lock: 100ns
*   Main Memory: 100ns
*   SSD Read: 150us
*   Disk Seek: 10ms
*   Round Trip (Datacenter): 500us
*   Round Trip (International): 150ms
*   **Key Insight:** Disk is 100x slower than Memory. Network is slow.

## Q3: How do you estimate QPS for a system with 10M DAU?
**Answer:**
1.  **Assume usage:** Each user makes 10 requests/day.
2.  **Total Requests:** $10M \times 10 = 100M$ requests/day.
3.  **Seconds in a day:** 86,400 $\approx 100,000$.
4.  **Average QPS:** $100M / 100K = 1,000$ QPS.
5.  **Peak QPS:** Usually $2\times$ or $3\times$ average. $\approx 2,000 - 3,000$ QPS.

## Q4: What is the difference between Throughput and Latency?
**Answer:**
*   **Throughput:** How much work the system can do per unit time (e.g., 1000 requests per second). Like the width of a pipe.
*   **Latency:** How long one specific request takes (e.g., 200ms). Like the length of the pipe.
*   **Relation:** You can have high throughput but high latency (batch processing). You can have low latency but low throughput (single thread). Ideally, we want High Throughput and Low Latency.

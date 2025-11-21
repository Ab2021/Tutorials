# Day 1 Deep Dive: Back-of-the-Envelope Calculations

## 1. Why Estimate?
In interviews and real life, you need to estimate capacity before building.
*   How many servers do we need?
*   How much storage for 5 years?
*   What is the network bandwidth requirement?

## 2. Latency Numbers Every Programmer Should Know (Jeff Dean)
| Operation | Time (Approx) |
| :--- | :--- |
| L1 Cache Reference | 0.5 ns |
| Branch Mispredict | 5 ns |
| L2 Cache Reference | 7 ns |
| Mutex Lock/Unlock | 100 ns |
| Main Memory Reference | 100 ns |
| Compress 1KB with Zippy | 10,000 ns (10 µs) |
| Send 2KB over 1Gbps Network | 20,000 ns (20 µs) |
| SSD Random Read | 150,000 ns (150 µs) |
| Read 1MB sequentially from Memory | 250,000 ns (250 µs) |
| Round Trip within same Data Center | 500,000 ns (500 µs) |
| Disk Seek | 10,000,000 ns (10 ms) |
| Read 1MB sequentially from Network | 10,000,000 ns (10 ms) |
| Read 1MB sequentially from Disk | 30,000,000 ns (30 ms) |
| Send Packet CA -> Netherlands -> CA | 150,000,000 ns (150 ms) |

**Takeaway:**
*   Memory is fast. Disk is slow. Network is unpredictable.
*   Avoid disk seeks (random I/O). Prefer sequential I/O.
*   Cache aggressively.

## 3. Power of Two (Approximations)
*   $2^{10} \approx 10^3$ (1 Thousand - KB)
*   $2^{20} \approx 10^6$ (1 Million - MB)
*   $2^{30} \approx 10^9$ (1 Billion - GB)
*   $2^{40} \approx 10^{12}$ (1 Trillion - TB)
*   $2^{50} \approx 10^{15}$ (1 Quadrillion - PB)

## 4. Estimation Example: Twitter Storage
**Scenario:** 300M Daily Active Users (DAU). 50% tweet once a day. Average tweet size 100 bytes (text) + metadata. 10% contain images (1MB).

**Calculations:**
1.  **Text Storage:**
    *   Tweets/Day = $300M \times 0.5 = 150M$ tweets.
    *   Size = $150M \times 100B = 15GB/day$.
2.  **Media Storage:**
    *   Images/Day = $150M \times 10\% = 15M$ images.
    *   Size = $15M \times 1MB = 15TB/day$.
3.  **Total Storage:**
    *   $\approx 15TB/day$.
    *   5 Years = $15TB \times 365 \times 5 \approx 27PB$.

**Conclusion:** You need a distributed blob storage (S3/HDFS), not a single MySQL instance.

## 5. QPS (Queries Per Second)
*   **DAU:** 300M.
*   **Reads:** Users view 100 tweets/day.
    *   $300M \times 100 = 30B$ reads/day.
    *   $30B / 86400 \approx 350,000$ QPS.
*   **Writes:** 150M tweets/day.
    *   $150M / 86400 \approx 1,700$ QPS.
*   **Ratio:** Read:Write ratio is ~200:1. Optimize for Reads (Caching!).

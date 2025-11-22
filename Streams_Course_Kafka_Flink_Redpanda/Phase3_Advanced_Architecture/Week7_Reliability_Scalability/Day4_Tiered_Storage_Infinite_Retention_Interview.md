# Day 4: Tiered Storage - Interview Prep

## Interview Questions & Challenges

### Common Interview Questions
1.  **Q: What is the main benefit of Tiered Storage?**
    -   *A*: Cost reduction (S3 is cheaper than SSD) and infinite retention without operational headache.

2.  **Q: Does Tiered Storage affect write latency?**
    -   *A*: No. Writes always go to local disk first. Upload happens asynchronously.

3.  **Q: How does it handle compaction?**
    -   *A*: Compaction usually happens locally before upload, or tiered storage engines support compacted topics in S3 (complex).

### Production Challenges
-   **Challenge**: **S3 Throttling**.
    -   *Scenario*: 503 Slow Down.
    -   *Fix*: Randomize object prefixes (S3 partitioning) or reduce parallelism of archiver.

### Troubleshooting Scenarios
**Scenario**: High egress cost.
-   *Cause*: Consumers reading cold data frequently (e.g., a buggy job restarting from 0 every hour).
-   *Fix*: Fix the consumer or increase local retention.

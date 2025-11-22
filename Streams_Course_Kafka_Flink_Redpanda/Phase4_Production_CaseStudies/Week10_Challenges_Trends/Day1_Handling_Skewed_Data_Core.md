# Day 1: Handling Skewed Data

## Core Concepts & Theory

### The Problem of Skew
In distributed systems, we assume data is evenly distributed. In reality, it follows a **Zipfian distribution** (Power Law).
-   **Example**: 80% of traffic comes from 20% of keys (e.g., "Justin Bieber" on Twitter, "iPhone" on Amazon).
-   **Impact**: One partition/task is overloaded (100% CPU), while others are idle. The whole job slows down to the speed of the slowest task.

### Types of Skew
1.  **Data Skew**: Some keys have more data than others.
2.  **Processing Skew**: Some records take longer to process (e.g., complex regex on large payload).

### Mitigation Strategies
1.  **Salting (Random Prefix)**: Add a random number (0-N) to the key. `Key` -> `Key_1`, `Key_2`. Distributes the hot key to N partitions.
2.  **Local Aggregation**: Pre-aggregate on the random key, then global aggregate.
3.  **Broadcast Join**: If one side is small, broadcast it to avoid shuffling the large skewed side.

### Architectural Reasoning
**Why not just "Auto-Scale"?**
Auto-scaling adds more workers. But if *one single key* has more data than *one single CPU* can handle, adding 1000 CPUs won't help. That key must go to one CPU (for correctness/ordering). You MUST break the key (Salting).

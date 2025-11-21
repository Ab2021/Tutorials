# Day 24 Interview Prep: Stream Processing

## Q1: Event Time vs Processing Time?
**Answer:**
*   **Event Time:** The time the user clicked the button. (Correctness).
*   **Processing Time:** The time the server received the packet. (Latency).
*   **Issue:** Network lag. An event from 12:00 might arrive at 12:05.
*   **Solution:** Use Watermarks to wait for late data.

## Q2: How does Flink guarantee Exactly-Once?
**Answer:**
*   **Checkpointing:** Flink periodically saves the state (e.g., Counts) and the Kafka Offset to a distributed file system (HDFS/S3).
*   **Chandy-Lamport Algorithm:** A distributed snapshot algorithm.
*   **Recovery:** On failure, restore state from last checkpoint and replay Kafka from saved offset.

## Q3: Lambda vs Kappa?
**Answer:**
*   **Lambda:** Batch + Stream. Complex. Good for legacy.
*   **Kappa:** Stream only. Simple. Good for modern stacks (Kafka + Flink).

## Q4: What is Backpressure in streaming?
**Answer:**
*   When the consumer (Flink) cannot keep up with the producer (Kafka).
*   **Mechanism:** Flink slows down reading from Kafka. Kafka buffers data on disk.
*   **Result:** Latency increases, but system doesn't crash.

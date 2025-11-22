# Day 4: Producers & Consumers

## Core Concepts & Theory

### The Producer
Producers write data to topics.
-   **Partitioning Strategy**: How does the producer decide which partition to send to?
    -   *Round-Robin*: If no key is provided.
    -   *Key-Hash*: If a key is provided (`hash(key) % num_partitions`). This ensures all events for the same key (e.g., `user_id`) go to the same partition (and thus are ordered).
-   **Batching**: Producers buffer messages to send them in batches for higher throughput.

### The Consumer
Consumers read data from topics.
-   **Pull Model**: Consumers pull data from brokers. This allows the consumer to control the rate (backpressure).
-   **Consumer Groups**: A set of consumers working together to consume a topic.
    -   Each partition is consumed by *only one* consumer in the group.
    -   This is the mechanism for **parallel consumption**.

### Architectural Reasoning
**Why Consumer Groups?**
If you have a topic with 1TB of data, a single consumer is too slow. You want to parallelize. Consumer Groups allow you to spin up N consumers, and Kafka automatically distributes the M partitions among them. If a consumer fails, Kafka performs a **Rebalance** to reassign its partitions to the survivors.

### Key Components
-   **ProducerRecord**: The object sent (Key, Value, Timestamp).
-   **ConsumerGroup**: Logical grouping for parallel processing.
-   **Rebalancing**: The process of redistributing partitions.

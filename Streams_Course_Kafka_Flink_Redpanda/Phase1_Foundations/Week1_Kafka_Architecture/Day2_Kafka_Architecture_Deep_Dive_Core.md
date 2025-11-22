# Day 2: Kafka Architecture Deep Dive

## Core Concepts & Theory

### The Broker
A **Broker** is a single Kafka server. It receives messages from producers, assigns them offsets, and commits them to storage on disk. It also services fetch requests from consumers.

### The Cluster
A **Cluster** is a group of brokers working together.
-   **Controller**: One broker is elected as the Controller. It manages the states of partitions and replicas and performs administrative tasks (like reassigning partitions).
-   **Metadata**: Information about where partitions exist. Clients (producers/consumers) cache this metadata to know which broker to talk to.

### Zookeeper vs. KRaft
-   **Zookeeper (Legacy)**: External service used for cluster coordination, leader election, and storing metadata. It was a bottleneck and operational burden.
-   **KRaft (Kafka Raft)**: The modern architecture (KIP-500). Metadata is stored in an internal Kafka topic (`@metadata`). The Controller is now a quorum of brokers using the Raft consensus algorithm. This removes the Zookeeper dependency.

### Architectural Reasoning
**Why Dumb Broker / Smart Client?**
Kafka pushes complexity to the client. The broker does not track which messages a consumer has read.
-   **Broker**: "Here is message 100 to 200." (Stateless-ish)
-   **Consumer**: "I have read up to 200." (Tracks its own state)
This design allows Kafka to scale massively because the broker does minimal work per consumer.

### Key Components
-   **Broker**: Storage and network layer.
-   **Controller**: Brain of the cluster.
-   **Zookeeper/KRaft**: Consensus and metadata store.

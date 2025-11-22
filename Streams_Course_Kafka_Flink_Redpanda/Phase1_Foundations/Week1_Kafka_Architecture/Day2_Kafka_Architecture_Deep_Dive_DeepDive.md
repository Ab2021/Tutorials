# Day 2: Kafka Architecture - Deep Dive

## Deep Dive & Internals

### The Controller's Role
The **Controller** is the "Brain".
-   It monitors broker liveness (via Zookeeper/KRaft heartbeats).
-   It elects partition leaders.
-   It tells other brokers: "You are now the leader for Partition 0."
-   **Split Brain**: If two brokers think they are the controller, chaos ensues. Generation IDs (Epochs) prevent this.

### KRaft (Kafka Raft) Internals
KRaft removes Zookeeper.
-   **Metadata Topic**: `@metadata` is an internal topic that stores the cluster state.
-   **Quorum**: A subset of brokers (Voters) participate in Raft consensus to elect the active Controller.
-   **Snapshotting**: To prevent the metadata log from growing forever, snapshots of the state are taken.

### Request Purgatory
When a request cannot be satisfied immediately (e.g., `acks=all` but replicas haven't acked yet), it is placed in the **Purgatory**.
-   It is a hierarchical timing wheel.
-   Requests sit there until the condition is met or they timeout.

### Advanced Reasoning
**Why is Zookeeper removal a big deal?**
-   **Scalability**: Zookeeper was a bottleneck for clusters with millions of partitions. KRaft scales much better.
-   **Simplicity**: One less distributed system to manage.
-   **Atomic Metadata**: Metadata updates are now atomic log appends.

### Performance Implications
-   **Controller Failover**: In Zookeeper, failover could take seconds/minutes (loading metadata). In KRaft, the standby controller already has the metadata in memory, so failover is near-instant.

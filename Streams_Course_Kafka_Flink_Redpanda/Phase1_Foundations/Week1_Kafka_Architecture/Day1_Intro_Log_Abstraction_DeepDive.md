# Day 1: The Log Abstraction - Deep Dive

## Deep Dive & Internals

### The Log as the Source of Truth
In a log-centric architecture, the **Log** is the single source of truth.
-   **Database as a Derived View**: A database table is just a cached view of the log. If you lose the database, you can rebuild it by replaying the log.
-   **Dual Writes Problem**: Writing to a DB and publishing an event is hard (two-phase commit?). With the Log, you write to the Log first, and the DB updates by consuming the Log (Change Data Capture).

### Sequential I/O vs Random I/O
The "Log" abstraction exploits the physics of disk drives.
-   **Random I/O**: Slow (seeking). 100-200 IOPS on HDD.
-   **Sequential I/O**: Fast. 100MB/s+ on HDD.
Kafka is designed to be "disk-bound" but acts like "network-bound" because disk sequential writes are so fast they often saturate the network before the disk.

### Zero-Copy Optimization
Kafka uses the Linux `sendfile` system call (Java `FileChannel.transferTo`).
-   **Traditional**: Disk -> Kernel Buffer -> User Space -> Kernel Socket Buffer -> NIC. (4 copies, 4 context switches).
-   **Zero-Copy**: Disk -> Kernel Buffer -> NIC. (2 copies, 2 context switches).
This is why Kafka can handle millions of messages per second with very low CPU usage.

### Advanced Reasoning
**Why not a Message Queue (RabbitMQ)?**
-   **RabbitMQ**: Smart broker, dumb consumer. Broker tracks state. Messages are removed after consumption. Good for complex routing.
-   **Kafka**: Dumb broker, smart consumer. Log is persisted. Good for high throughput, replayability, and massive scale.

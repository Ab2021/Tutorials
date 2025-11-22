# Day 4: Operator State & Broadcast

## Core Concepts & Theory

### Operator State
State bound to a parallel task instance, not a key.
-   **ListState**: A list of items. On rescale, can be:
    -   **Redistributed**: Round-robin.
    -   **Union**: Broadcast to all (everyone gets everything).

### Broadcast State
A special type of Operator State.
-   **Pattern**: One low-throughput "Control Stream" (Rules) is broadcast to all instances of a high-throughput "Data Stream".
-   **Storage**: Replicated on every node.
-   **Usage**: Dynamic Rules, Feature Flags, Lookup Tables.

### Architectural Reasoning
**Why Broadcast?**
Imagine a "Fraud Detection" job. You have 1000 rules. You want to update rules dynamically without restarting.
-   Stream 1: Transactions (Keyed by User).
-   Stream 2: Rules (Broadcast).
-   `connect(rules).process()`: The process function has access to the "current rules" in Broadcast State.

### Key Components
-   `BroadcastProcessFunction`: The function to handle connected streams.
-   `MapStateDescriptor`: Broadcast state is always a Map.

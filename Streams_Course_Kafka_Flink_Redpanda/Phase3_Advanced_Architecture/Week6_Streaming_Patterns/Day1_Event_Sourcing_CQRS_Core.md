# Day 1: Event Sourcing & CQRS

## Core Concepts & Theory

### Event Sourcing
Instead of storing the *current state* of an entity, store the *sequence of events* that led to that state.
-   **State**: Derived by replaying events.
-   **Immutability**: Events are facts. They cannot be changed, only compensated (e.g., "OrderCreated", "OrderCancelled").

### CQRS (Command Query Responsibility Segregation)
Splitting the model into:
-   **Command Side (Write)**: Validates commands and emits events. High consistency.
-   **Query Side (Read)**: Consumes events and builds materialized views (e.g., SQL, ElasticSearch). High availability/scalability.

### Architectural Reasoning
**Why Event Sourcing?**
-   **Audit Trail**: You have a perfect history of "who did what and when".
-   **Time Travel**: You can reconstruct the state of the system at any point in time.
-   **Debuggability**: Copy the production event log to dev and replay it to reproduce a bug.

### Key Components
-   **Event Store**: Kafka/Redpanda is the perfect Event Store (durable, ordered).
-   **Projection**: A Flink job that consumes events and updates a Read Model (DB).

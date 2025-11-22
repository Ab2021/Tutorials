# Day 4: Streaming Databases - Deep Dive

## Deep Dive & Internals

### Incremental View Maintenance (IVM)
How to update `SELECT sum(sales) FROM orders` without rescanning the table?
-   **New Event**: `+10`.
-   **Old Sum**: `500`.
-   **New Sum**: `510`.
-   **Join**: `A JOIN B`. If `A` changes, look up matching `B` in index, emit change.

### Consistency Models
-   **Eventual Consistency**: You might see old data.
-   **Strong Consistency**: You see the correct answer as of a specific timestamp.
-   **Materialize**: Offers "Consistency" (all views update atomically for a timestamp).

### Pull vs Push Queries
-   **Push Query**: "Tell me whenever the result changes". (Subscription / WebSocket).
-   **Pull Query**: "Tell me the current result". (Request / Response).

### Performance Implications
-   **Join State**: Maintaining a join requires storing both tables in state. Expensive.
-   **Churn**: High update rate = High CPU to maintain the view.

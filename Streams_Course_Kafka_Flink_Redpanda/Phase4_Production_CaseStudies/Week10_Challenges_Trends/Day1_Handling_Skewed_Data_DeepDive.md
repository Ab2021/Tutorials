# Day 1: Skew - Deep Dive

## Deep Dive & Internals

### Two-Phase Aggregation (Salting)
**Scenario**: Count views for "Justin Bieber" (1M/sec).
**Phase 1 (Local)**:
-   Add Salt: `JustinBieber` -> `JustinBieber_0` ... `JustinBieber_9`.
-   KeyBy: `SaltedKey`.
-   Count: `JustinBieber_0: 100k`, `JustinBieber_1: 100k`...
**Phase 2 (Global)**:
-   KeyBy: Original Key (`JustinBieber`).
-   Sum: `100k + 100k ... = 1M`.
**Result**: The heavy lifting (counting 1M events) is spread across 10 tasks. The final aggregation only sums 10 numbers.

### Skew in Joins
**Scenario**: Join `Orders` (Skewed) with `Users`.
-   **Regular Join**: Both sides shuffled by UserID. Skewed task dies.
-   **Broadcast Join**: Broadcast `Users` to all tasks. `Orders` stays local (no shuffle).
-   **Salted Join**: Salt `Orders` (0-9). Replicate `Users` 10 times (`User_0`...`User_9`). Join `Order_N` with `User_N`.

### Performance Implications
-   **Network**: Salting increases network shuffle (if not careful).
-   **Memory**: Replicating the small side (in Salted Join) increases memory usage.

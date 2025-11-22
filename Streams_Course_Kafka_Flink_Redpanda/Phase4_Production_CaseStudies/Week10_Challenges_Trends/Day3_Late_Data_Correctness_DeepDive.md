# Day 3: Late Data - Deep Dive

## Deep Dive & Internals

### Watermark Strategies
1.  **Monotonous**: Timestamps always increase. (Rare).
2.  **Bounded Out of Orderness**: `Watermark = MaxTimestamp - Delay`.
    -   *Delay*: How much lateness we tolerate before triggering.
    -   *Example*: Delay = 5s. If we see T=100, W=95. We are ready to close window [0-90].

### Idleness
If a partition has no data, it sends no watermarks. The global watermark (Min of all partitions) stalls.
-   **Impact**: Windows never close. Downstream waits forever.
-   **Fix**: `withIdleness(Duration.ofMinutes(1))`. If no data for 1m, mark partition as idle (ignore it for watermark calculation).

### Retractions
If we emit a result "Count=5", and a late event arrives, the count becomes 6.
-   **Append Stream**: Emits `5`, then `6`. (Downstream must know `6` replaces `5`).
-   **Retract Stream**: Emits `5`, then `-5` (Undo), then `6`. (Standard for SQL).
-   **Upsert Stream**: Emits `Key=X, Val=6`. (Overwrites).

### Performance Implications
-   **State Size**: `allowedLateness` keeps windows in memory. Large lateness = Huge state.

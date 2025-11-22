# Day 5: Triggers - Deep Dive

## Deep Dive & Internals

### FIRE vs PURGE
-   **FIRE**: Call the window function, keep the state. (Good for early updates).
-   **PURGE**: Clear the state.
-   **FIRE_AND_PURGE**: Emit result and clear. (Standard for tumbling windows).

### ContinuousProcessingTimeTrigger
Fires periodically. Useful for "speculative" results.
-   e.g., "Show me the current top 10 trending items every 5 seconds, even though the 1-hour window isn't done."

### Advanced Reasoning
**The "Global Window" Trap**
A `GlobalWindow` never ends. The default trigger never fires. You *must* attach a custom trigger (like `CountTrigger`) or it will just accumulate state until OOM.

### Performance Implications
-   **Early Firing**: Increases downstream load (more updates).
-   **Evictors**: Force the use of `ProcessWindowFunction` (state must be kept to allow eviction). Prevents incremental aggregation optimization.

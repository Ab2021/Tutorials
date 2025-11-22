# Day 4: Windowing Strategies

## Core Concepts & Theory

### Types of Windows
1.  **Tumbling**: Fixed size, non-overlapping. (e.g., "Every 5 minutes").
2.  **Sliding**: Fixed size, overlapping. (e.g., "Last 10 mins, updated every 1 min").
3.  **Session**: Dynamic size. Defined by a gap of inactivity. (e.g., "User session ends after 30 mins idle").
4.  **Global**: One giant window. Requires a custom trigger to fire.

### Window Lifecycle
-   **Creation**: When the first element for a key/window arrives.
-   **Accumulation**: Elements are added to the window state.
-   **Firing**: Trigger decides to compute the result.
-   **Purging**: Clearing the content.

### Architectural Reasoning
**Why Windows need State?**
To calculate "average price over 1 hour", Flink must store either all prices (ListState) or the running sum/count (ReducingState) until the hour is up. This state is managed by the State Backend.

### Key Components
-   **WindowAssigner**: Assigns element to window(s).
-   **Trigger**: When to evaluate.
-   **Evictor**: Which elements to keep (optional).
-   **WindowFunction**: The computation (Reduce, Aggregate, Process).

# Day 4: Windowing - Deep Dive

## Deep Dive & Internals

### Window State
-   **Incremental Aggregation** (`ReduceFunction`, `AggregateFunction`): Computes as data arrives. Stores only 1 value (e.g., sum). Efficient.
-   **Full Window Function** (`ProcessWindowFunction`): Stores ALL elements until trigger. Expensive (RAM/State), but allows access to metadata (start/end time) and iterating over all elements.
-   **Combined**: You can use `window.aggregate(Agg, Process)` to get efficiency + metadata.

### Session Window Implementation
Session windows are hard because they merge.
-   Initially, every element creates a new session window `[t, t+gap]`.
-   If two windows overlap, they are **merged** into a larger window.
-   State must be merged.

### Advanced Reasoning
**Sliding Window Optimization**
A sliding window (size 1hr, slide 1min) assigns every element to 60 windows! This explodes state.
-   **Optimization**: Panes (tumbling windows of GCD size) or slicing. Flink's default implementation actually duplicates the data into multiple buckets. Be careful with small slides.

### Performance Implications
-   **Large Windows**: A 24-hour window with full process function will blow up memory. Use incremental aggregation.

# Day 5: Triggers & Evictors

## Core Concepts & Theory

### Triggers
A **Trigger** determines when a window is ready to be processed.
-   **EventTimeTrigger**: Fires when Watermark passes window end. (Default).
-   **ProcessingTimeTrigger**: Fires based on wall-clock time.
-   **CountTrigger**: Fires when N elements arrive.
-   **PurgingTrigger**: Fires and then clears the window.

### Evictors
An **Evictor** can remove elements from the window *before* or *after* the trigger fires.
-   **CountEvictor**: Keep only last N elements.
-   **DeltaEvictor**: Keep elements based on a delta threshold.

### Architectural Reasoning
**Custom Triggers**
You might want a window that fires "Every 1 minute OR when 1000 items arrive" (Early firing). This allows low latency updates for a long window.

### Key Components
-   `onElement()`: Called for every record.
-   `onEventTime()`: Called when watermark passes.
-   `onProcessingTime()`: Called when system timer fires.

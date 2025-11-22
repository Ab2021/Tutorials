# Lab 14: Async I/O

## Difficulty
🔴 Hard

## Estimated Time
60 mins

## Learning Objectives
-   Use Async I/O to call external APIs without blocking.

## Problem Statement
*Note: PyFlink support for Async I/O is limited compared to Java. We will simulate the concept or use a ThreadPool.*
Simulate an external API call that takes 1s. Use `map` vs `async_wait` (conceptual).

## Starter Code
```python
# PyFlink Async I/O is complex. 
# We will focus on the concept:
# OrderedWait vs UnorderedWait
```

## Hints
<details>
<summary>Hint 1</summary>
If you block in a `map` function, you block the checkpointing barrier too!
</details>

## Solution
<details>
<summary>Click to reveal solution</summary>

*Conceptual Solution (Java syntax is standard for Async I/O)*:
```java
AsyncDataStream.unorderedWait(
    stream,
    new AsyncDatabaseRequest(),
    1000, TimeUnit.MILLISECONDS,
    100);
```
In PyFlink, ensure you use a thread pool inside your map function if you must do blocking I/O, but true Async I/O requires the Async operator.
</details>
